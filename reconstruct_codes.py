#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Reconstruct discrete audio codes from Voxtral TTS voice embeddings.

Voice embeddings (voice_embedding/{id}.pt) are pre-summed LLM-space vectors:

    voice_emb[t] = Σ_{k=0}^{36}  E[ code_k[t] + offset_k ]

where E is the shared audio_token_embedding table (shape [9088, 3072]).

This script inverts that sum via **coordinate descent**:
  - Fix all codes except one → the optimal update is a nearest-neighbour
    search over that codebook's slice of E (residual = v - sum of others).
  - Iterate over all 37 codebooks until no code changes.

In 3072-D with only 9088 total vocab entries the embedding rows are
effectively linearly independent, so coordinate descent converges to the
exact solution in 1–3 full passes.

Usage
-----
# Load embedding table from a safetensors checkpoint shard:
python reconstruct_codes.py \
    --embedding-weight  /path/to/model/consolidated.safetensors \
    --voice-embedding   /path/to/voice_embeddings/en_female.pt \
    --output            en_female_codes.pt

# Load from a raw .bin checkpoint:
python reconstruct_codes.py \
    --embedding-weight  /path/to/model/pytorch_model.bin \
    --voice-embedding   /path/to/voice_embeddings/neutral_male.pt \
    --output            neutral_male_codes.pt

# Reconstruct ALL voice embeddings in a folder:
python reconstruct_codes.py \
    --embedding-weight  /path/to/model/consolidated.safetensors \
    --voice-dir         /path/to/voice_embeddings \
    --output-dir        ./reconstructed_codes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codebook layout (must match MultimodalAudioModelArgs defaults)
# ---------------------------------------------------------------------------

N_SPECIAL_TOKENS = 2      # AudioSpecialTokens: EMPTY=0, END=1
SEMANTIC_VOCAB   = 8192   # real semantic codes
ACOUSTIC_VOCAB   = 21     # FSQ levels per acoustic codebook
N_ACOUSTIC_CBS   = 36     # number of acoustic codebooks
N_CODEBOOKS      = 1 + N_ACOUSTIC_CBS  # 37 total

# Padded per-codebook sizes stored in MultiVocabEmbeddings
# (include_special_tokens=True, pad_to_multiple=None)
_CODEBOOK_SIZES = [SEMANTIC_VOCAB + N_SPECIAL_TOKENS] + \
                  [ACOUSTIC_VOCAB + N_SPECIAL_TOKENS] * N_ACOUSTIC_CBS
# [8194, 23, 23, ...(36)]

OFFSETS: torch.Tensor = torch.tensor(
    np.cumsum([0] + _CODEBOOK_SIZES[:-1]), dtype=torch.long
)  # shape [37]


# ---------------------------------------------------------------------------
# Embedding-table loader
# ---------------------------------------------------------------------------

# Key names used in different checkpoint formats
_EMBTABLE_KEYS = [
    "audio_token_embedding.embeddings.weight",                         # post-remap (our model.py)
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", # raw Mistral checkpoint
]


def load_embedding_table(weight_path: str | Path) -> torch.Tensor:
    """Load the audio_token_embedding weight from a checkpoint file.

    Supports .safetensors and .pt/.bin files.

    Returns:
        Tensor of shape [padded_vocab, hidden_size] (float32 on CPU).
    """
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    logger.info("Loading embedding table from %s …", path)

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file, safe_open
        except ImportError:
            raise ImportError("pip install safetensors")

        # Try loading only the needed key to save memory
        with safe_open(str(path), framework="pt", device="cpu") as f:
            available = list(f.keys())
            for key in _EMBTABLE_KEYS:
                if key in available:
                    table = f.get_tensor(key).float()
                    logger.info("Found '%s', shape %s", key, table.shape)
                    return table

        # Key not found in this shard — check all keys for a partial match
        logger.warning(
            "Audio embedding key not found in %s.\n"
            "Available keys containing 'embed': %s",
            path,
            [k for k in available if "embed" in k.lower()][:20],
        )
        raise KeyError(
            f"Could not find audio embedding table in {path}. "
            f"Expected one of: {_EMBTABLE_KEYS}"
        )

    else:
        # .pt / .bin
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        state = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
        for key in _EMBTABLE_KEYS:
            if key in state:
                table = state[key].float()
                logger.info("Found '%s', shape %s", key, table.shape)
                return table
        embed_keys = [k for k in state if "embed" in k.lower()][:20]
        raise KeyError(
            f"Could not find audio embedding table.\n"
            f"Keys containing 'embed': {embed_keys}"
        )


# ---------------------------------------------------------------------------
# Core inversion: coordinate descent
# ---------------------------------------------------------------------------

def reconstruct_codes(
    voice_emb: torch.Tensor,  # [T, D]
    emb_table: torch.Tensor,  # [V, D]
    offsets: torch.Tensor,    # [K]  per-codebook start indices in emb_table
    codebook_sizes: list[int],
    max_iters: int = 10,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Recover discrete codes from a summed embedding via coordinate descent.

    Args:
        voice_emb:      [T, D]  summed voice embedding (one row per frame).
        emb_table:      [V, D]  the shared MultiVocabEmbeddings weight matrix.
        offsets:        [K]     start index of each codebook in emb_table.
        codebook_sizes: list[K] number of entries per codebook.
        max_iters:      maximum coordinate-descent passes.
        device:         computation device.

    Returns:
        codes: [T, K] int64 tensor.
            Row t, column k = discrete code for frame t, codebook k.
            These are the **raw** codes (0 = EMPTY, 1 = END, 2+ = real code).
    """
    voice_emb = voice_emb.to(device=device, dtype=torch.float32)
    emb_table = emb_table.to(device=device, dtype=torch.float32)
    offsets = offsets.to(device)

    T, D = voice_emb.shape
    K = len(codebook_sizes)

    # ------------------------------------------------------------------
    # Initialise: greedy nearest-neighbour for each codebook independently
    # (ignoring cross-codebook interactions as a warm start)
    # ------------------------------------------------------------------
    codes = torch.zeros(T, K, dtype=torch.long, device=device)
    current_sum = torch.zeros(T, D, device=device)

    for k in range(K):
        start = int(offsets[k].item())
        size  = codebook_sizes[k]
        cb_embs = emb_table[start : start + size]          # [size, D]
        # residual for this codebook (ignoring others at init)
        residual = voice_emb - (current_sum + voice_emb)   # = zero as warm-start
        # nearest-neighbour in the full table slice
        dists = torch.cdist(voice_emb.unsqueeze(0), cb_embs.unsqueeze(0)).squeeze(0)  # [T, size]
        codes[:, k] = dists.argmin(dim=-1)
        current_sum += cb_embs[codes[:, k]]                # add selected embeddings

    # ------------------------------------------------------------------
    # Coordinate descent
    # ------------------------------------------------------------------
    for iteration in range(max_iters):
        n_changed = 0

        for k in range(K):
            start    = int(offsets[k].item())
            size     = codebook_sizes[k]
            cb_embs  = emb_table[start : start + size]     # [size, D]

            # Remove current codebook k contribution
            old_emb  = cb_embs[codes[:, k]]                # [T, D]
            residual = voice_emb - (current_sum - old_emb) # [T, D]

            # Nearest-neighbour over this codebook's slice
            # residual[t] should equal cb_embs[c_k[t]] if all others are exact
            dists     = torch.cdist(
                residual.unsqueeze(0), cb_embs.unsqueeze(0)
            ).squeeze(0)                                    # [T, size]
            new_codes = dists.argmin(dim=-1)               # [T]

            changed = (new_codes != codes[:, k]).sum().item()
            n_changed += changed

            # Update running sum
            current_sum = current_sum - old_emb + cb_embs[new_codes]
            codes[:, k] = new_codes

        logger.debug("Iteration %d: %d codes changed", iteration + 1, n_changed)
        if n_changed == 0:
            logger.info("Converged after %d iteration(s).", iteration + 1)
            break
    else:
        logger.warning("Did not fully converge in %d iterations.", max_iters)

    return codes  # [T, K]


def reconstruction_error(
    voice_emb: torch.Tensor,  # [T, D]
    codes: torch.Tensor,      # [T, K]
    emb_table: torch.Tensor,  # [V, D]
    offsets: torch.Tensor,    # [K]
) -> dict[str, float]:
    """Compute re-embedding error: how well codes reproduce voice_emb."""
    voice_emb = voice_emb.float()
    emb_table = emb_table.float()
    offsets   = offsets.to(codes.device)

    recon = torch.zeros_like(voice_emb)
    for k in range(codes.shape[1]):
        start  = int(offsets[k].item())
        recon += emb_table[codes[:, k] + start]

    diff  = voice_emb - recon
    mae   = diff.abs().mean().item()
    rmse  = diff.pow(2).mean().sqrt().item()
    max_e = diff.abs().max().item()
    rel   = (diff.norm() / voice_emb.norm()).item()

    return {"MAE": mae, "RMSE": rmse, "MaxAbsErr": max_e, "RelErr": rel}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct discrete audio codes from Voxtral TTS voice embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embedding-weight", required=True, metavar="PATH",
        help="Path to a checkpoint file (.safetensors or .pt/.bin) that contains "
             "the audio_token_embedding table.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--voice-embedding", metavar="PT_FILE",
        help="Path to a single voice embedding .pt file.",
    )
    src.add_argument(
        "--voice-dir", metavar="DIR",
        help="Directory of voice embedding .pt files — reconstruct all of them.",
    )

    dst = p.add_mutually_exclusive_group()
    dst.add_argument(
        "--output", metavar="PT_FILE",
        help="Output path for the reconstructed codes tensor (single file mode).",
    )
    dst.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory for reconstructed code tensors (batch mode).",
    )

    p.add_argument("--max-iters", type=int, default=10,
                   help="Maximum coordinate-descent iterations.")
    p.add_argument("--device", default="cpu",
                   help="Torch device ('cpu' or 'cuda').")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def reconstruct_file(
    voice_path: Path,
    emb_table: torch.Tensor,
    max_iters: int,
    device: str,
) -> torch.Tensor:
    """Reconstruct codes for a single voice embedding file. Returns [T, 37]."""
    voice_emb: torch.Tensor = torch.load(
        str(voice_path), map_location="cpu", weights_only=True
    ).float()
    T, D = voice_emb.shape
    logger.info("  %s: shape [%d, %d]", voice_path.name, T, D)

    codes = reconstruct_codes(
        voice_emb=voice_emb,
        emb_table=emb_table,
        offsets=OFFSETS,
        codebook_sizes=_CODEBOOK_SIZES,
        max_iters=max_iters,
        device=device,
    )

    err = reconstruction_error(voice_emb, codes.cpu(), emb_table.cpu(), OFFSETS)
    logger.info(
        "  Reconstruction error — MAE: %.2e  RMSE: %.2e  MaxAbs: %.2e  RelErr: %.4f",
        err["MAE"], err["RMSE"], err["MaxAbsErr"], err["RelErr"],
    )
    if err["RelErr"] > 1e-3:
        logger.warning(
            "  RelErr=%.4f is high — reconstruction may be approximate. "
            "Try --max-iters %d", err["RelErr"], max_iters * 2
        )

    # Print a small summary of the recovered codes
    _print_code_summary(codes, voice_path.stem)
    return codes


def _print_code_summary(codes: torch.Tensor, name: str) -> None:
    """Print a human-readable summary of the reconstructed codes."""
    T, K = codes.shape
    print(f"\n{'─'*60}")
    print(f"Voice: {name}  |  {T} frames  ({T/12.5:.2f}s @ 12.5Hz)")
    print(f"{'─'*60}")
    print(f"  Codes shape:    {list(codes.shape)}  (T frames × {K} codebooks)")
    print(f"  Semantic  (cb0): {codes[:, 0].tolist()[:10]} {'...' if T>10 else ''}")
    print(f"  Acoustic0 (cb1): {codes[:, 1].tolist()[:10]} {'...' if T>10 else ''}")
    sem = codes[:, 0]
    print(f"  Semantic range: [{sem.min().item()}, {sem.max().item()}]  "
          f"(0=EMPTY, 1=END, 2..={SEMANTIC_VOCAB+1})")
    aco = codes[:, 1:]
    print(f"  Acoustic range: [{aco.min().item()}, {aco.max().item()}]  "
          f"(0=EMPTY, 1=END, 2..={ACOUSTIC_VOCAB+1})")
    n_end = (sem == 1).sum().item()
    print(f"  END_AUDIO frames: {n_end}")
    print(f"{'─'*60}\n")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load the embedding table once
    emb_table = load_embedding_table(args.embedding_weight)
    logger.info("Embedding table loaded: shape %s", list(emb_table.shape))

    # ---- Single file ----
    if args.voice_embedding:
        voice_path = Path(args.voice_embedding)
        codes = reconstruct_file(voice_path, emb_table, args.max_iters, args.device)

        out = Path(args.output) if args.output else voice_path.with_suffix("_codes.pt")
        torch.save(codes, str(out))
        print(f"Saved codes [{codes.shape[0]}, {codes.shape[1]}] → {out}")

    # ---- Batch mode ----
    else:
        voice_dir  = Path(args.voice_dir)
        out_dir    = Path(args.output_dir) if args.output_dir else voice_dir / "codes"
        out_dir.mkdir(parents=True, exist_ok=True)

        pt_files = sorted(voice_dir.glob("*.pt"))
        if not pt_files:
            logger.error("No .pt files found in %s", voice_dir)
            sys.exit(1)

        for voice_path in pt_files:
            codes = reconstruct_file(voice_path, emb_table, args.max_iters, args.device)
            out = out_dir / voice_path.name
            torch.save(codes, str(out))
            print(f"  → {out}")

        print(f"\nAll {len(pt_files)} files written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
