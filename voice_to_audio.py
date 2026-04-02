#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Decode Voxtral TTS voice embeddings back to audio waveforms.

Pipeline
--------
voice_emb [T, 3072]
    │  coordinate descent over MultiVocabEmbeddings table
    ▼
codes [T, 37]   (0=EMPTY, 1=END_AUDIO, 2+=real codes)
    │  MistralAudioCodebook.decode() + transformer decoder
    ▼
waveform [samples]  @ 24 kHz

The last frame of every voice embedding is always the END_AUDIO marker
(confirmed: norm ≈ 14.75 vs ≈ 4–5 for real frames), so we get
(T-1) / 12.5 seconds of reconstructed audio per file.

Usage
-----
# Single file:
python -m voxtral_tts_pure.voice_to_audio \\
    --model   /path/to/Voxtral-4B-TTS-2603 \\
    --voice   voice_embeddings/neutral_female.pt \\
    --output  neutral_female.wav

# Entire folder:
python -m voxtral_tts_pure.voice_to_audio \\
    --model      /path/to/Voxtral-4B-TTS-2603 \\
    --voice-dir  voice_embeddings/ \\
    --output-dir reconstructed_audio/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

from audio_tokenizer import VoxtralTTSAudioTokenizer
from reconstruct_codes import (
    OFFSETS,
    _CODEBOOK_SIZES,
    load_embedding_table,
    reconstruct_codes,
    reconstruction_error,
    _print_code_summary,
)

logger = logging.getLogger(__name__)

SAMPLING_RATE = 24_000
FRAME_RATE    = 12.5   # Hz  (1 frame = 80 ms)


# ---------------------------------------------------------------------------
# Default Voxtral TTS codec config (matches mistralai/Voxtral-4B-TTS-2603)
# ---------------------------------------------------------------------------

_DEFAULT_CODEC_ARGS = {
    "channels": 1,
    "sampling_rate": 24000,
    "pretransform_patch_size": 240,
    "patch_proj_kernel_size": 7,
    "semantic_codebook_size": 8192,
    "semantic_dim": 256,
    "acoustic_codebook_size": 21,
    "acoustic_dim": 36,
    "conv_weight_norm": True,
    "causal": True,
    "attn_sliding_window_size": 16,
    "half_attn_window_upon_downsampling": True,
    "dim": 1024,
    "hidden_dim": 4096,
    "head_dim": 128,
    "n_heads": 8,
    "n_kv_heads": 8,
    "qk_norm_eps": 1e-6,
    "qk_norm": True,
    "use_biases": False,
    "norm_eps": 1e-2,
    "layer_scale": True,
    "layer_scale_init": 0.01,
    "encoder_transformer_lengths_str": "2,2,2,2",
    "encoder_convs_kernels_str":       "4,4,4,3",
    "encoder_convs_strides_str":       "2,2,2,1",
    "decoder_transformer_lengths_str": "2,2,2,2",
    "decoder_convs_kernels_str":       "3,4,4,4",
    "decoder_convs_strides_str":       "1,2,2,2",
}

_DEFAULT_AUDIO_MODEL_ARGS = {
    "semantic_codebook_size": 8192,
    "acoustic_codebook_size": 21,
    "n_acoustic_codebook": 36,
    "acoustic_transformer_args": {
        "input_dim": 3072,
        "dim": 3072,
        "n_layers": 3,
        "head_dim": 128,
        "hidden_dim": 9216,
        "n_heads": 32,
        "n_kv_heads": 8,
    },
}


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

# Key prefixes used in different checkpoint formats
_TOKENIZER_PREFIXES = (
    "audio_tokenizer.",
    "mm_audio_embeddings.audio_codebook_embeddings.",
)

_EMBTABLE_KEYS = (
    "audio_token_embedding.embeddings.weight",
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
)


def _iter_checkpoint_tensors(model_path: Path):
    """Yield (name, tensor) pairs from all checkpoint shards in model_path."""
    shard_files = sorted(model_path.glob("*.safetensors"))
    if shard_files:
        try:
            from safetensors.torch import safe_open
        except ImportError:
            raise ImportError("pip install safetensors")
        for sf in shard_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        return

    pt_files = sorted(model_path.glob("*.bin")) or sorted(model_path.glob("pytorch_model*.pt"))
    if pt_files:
        for pt in pt_files:
            shard = torch.load(str(pt), map_location="cpu", weights_only=True)
            yield from shard.items()
        return

    consol = model_path / "consolidated.safetensors"
    if consol.exists():
        from safetensors.torch import safe_open
        with safe_open(str(consol), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)
        return

    raise FileNotFoundError(f"No weight files found in {model_path}")


def load_decoder_and_embtable(
    model_path: Path,
    dtype: torch.dtype,
    device: str | torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    """Load VoxtralTTSAudioTokenizer (decoder weights only) and the embedding table.

    Returns:
        tokenizer:  VoxtralTTSAudioTokenizer with decoder + quantizer weights loaded.
        emb_table:  [V, D] float32 CPU tensor (MultiVocabEmbeddings weight).
    """

    # Try to get codec_args and audio_model_args from config
    codec_args      = _DEFAULT_CODEC_ARGS
    audio_model_args = _DEFAULT_AUDIO_MODEL_ARGS

    for cfg_name in ("params.json", "config.json"):
        cfg_path = model_path / cfg_name
        if cfg_path.exists():
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            mm = cfg.get("multimodal", cfg.get("audio_config", {}))
            if mm.get("audio_tokenizer_args") or mm.get("codec_args"):
                codec_args = mm.get("audio_tokenizer_args") or mm.get("codec_args")
                logger.info("Loaded codec_args from %s", cfg_name)
            if mm.get("audio_model_args"):
                audio_model_args = mm["audio_model_args"]
                at = audio_model_args.setdefault("acoustic_transformer_args", {})
                at.setdefault("input_dim", 3072)
                logger.info("Loaded audio_model_args from %s", cfg_name)
            break

    # Determine text hidden size from config (for embedding table dim)
    hidden_size = 3072  # Voxtral-4B default

    logger.info("Instantiating VoxtralTTSAudioTokenizer …")
    tokenizer = VoxtralTTSAudioTokenizer(
        codec_args=codec_args,
        audio_model_args=audio_model_args,
        text_hidden_size=hidden_size,
    )

    # ---- Scan checkpoint and load relevant weights ----
    emb_table: torch.Tensor | None = None
    n_tok = 0

    logger.info("Scanning checkpoint shards in %s …", model_path)
    for raw_name, tensor in _iter_checkpoint_tensors(model_path):

        # Embedding table
        if raw_name in _EMBTABLE_KEYS:
            emb_table = tensor.float().cpu()
            logger.info("  Found embedding table: %s  %s", raw_name, list(tensor.shape))
            continue

        # Audio tokenizer decoder weights
        sub = None
        if raw_name.startswith("audio_tokenizer."):
            sub = raw_name[len("audio_tokenizer."):]
        elif raw_name.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
            # already handled above as emb_table, but catch remainder if any
            sub = raw_name[len("mm_audio_embeddings.audio_codebook_embeddings."):]

        if sub is not None:
            try:
                tokenizer.load_weight(sub, tensor)
                n_tok += 1
            except Exception as exc:
                logger.debug("  Skipped %s: %s", sub, exc)

    if emb_table is None:
        raise RuntimeError(
            "Embedding table not found in checkpoint. "
            f"Expected one of: {_EMBTABLE_KEYS}"
        )
    if n_tok == 0:
        raise RuntimeError(
            "No audio tokenizer weights loaded. "
            "Check that the checkpoint contains 'audio_tokenizer.*' keys."
        )

    logger.info("Loaded %d audio tokenizer weight tensors.", n_tok)
    tokenizer = tokenizer.to(dtype=dtype, device=device).eval()
    return tokenizer, emb_table


# ---------------------------------------------------------------------------
# Single-file decode
# ---------------------------------------------------------------------------

def get_codes(voice_emb, emb_table, max_iters, device):
    codes = reconstruct_codes(
        voice_emb=voice_emb,
        emb_table=emb_table,
        offsets=OFFSETS,
        codebook_sizes=_CODEBOOK_SIZES,
        max_iters=max_iters,
        device=device,
    )  # [T, 37]
    return codes

def voice_embedding_to_audio(
    tokenizer: nn.Module,
    emb_table: torch.Tensor,
    max_iters: int = 10,
    voice_path: Path | None = None,
    voice_emb: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    codes: Any | None = None
) -> tuple[torch.Tensor, float]:
    """Decode one voice embedding file to a waveform.

    Returns:
        waveform:  1-D float32 CPU tensor of audio samples at 24 kHz.
        duration:  audio duration in seconds.
    """
    if voice_emb is None:
        assert voice_path, "Voice path cannot be None if no voice embedding provided"
        voice_emb: torch.Tensor = torch.load(
            str(voice_path), map_location="cpu", weights_only=True
        ).float()
    T, D = voice_emb.shape

    # ---- Step 1: reconstruct discrete codes ----
    logger.info("[%s]  Reconstructing codes from [%d, %d] embedding …",
                voice_path.name if voice_path else "", T, D)
    if codes is None:
        codes = reconstruct_codes(
            voice_emb=voice_emb,
            emb_table=emb_table,
            offsets=OFFSETS,
            codebook_sizes=_CODEBOOK_SIZES,
            max_iters=max_iters,
            device=device,
        )  # [T, 37]

    try:
        err = reconstruction_error(voice_emb, codes.cpu(), emb_table.cpu(), OFFSETS)
        logger.info(
            "[%s]  Code reconstruction — MAE: %.2e  RMSE: %.2e  RelErr: %.4f",
            voice_path.name if voice_path else "", err["MAE"], err["RMSE"], err["RelErr"],
        )
        if err["RelErr"] > 1e-3:
            logger.warning(
                "[%s]  RelErr %.4f is high — audio quality may be degraded.",
                voice_path.name if voice_path else "", err["RelErr"],
            )
    except:
        logger.warning(
            "Skipping reconstruction check"
        )

    _print_code_summary(codes, voice_path.stem if voice_path else "")

    # ---- Step 2: decode codes → waveform ----
    # decode_helper_batch_async expects [T, K] tensors; it will:
    #   • find END_AUDIO (codes[:, 0] == 1) and cut there
    #   • subtract 2 (strip AudioSpecialTokens offset)
    #   • run the codec decoder
    logger.info("[%s]  Decoding codes to waveform …", voice_path.name if voice_path else "")

    codes_dev = codes.to(device)
    with torch.inference_mode():
        waveforms = tokenizer.decode_helper_batch_async([codes_dev])

    waveform = waveforms[0].cpu().float()
    n_real_frames = T - 1           # last frame = END_AUDIO
    duration = n_real_frames / FRAME_RATE
    actual_samples = len(waveform)
    actual_dur = actual_samples / SAMPLING_RATE

    logger.info(
        "[%s]  Done — expected %.2fs (%d frames), got %.2fs (%d samples)",
        voice_path.name if voice_path else "", duration, n_real_frames, actual_dur, actual_samples,
    )
    return waveform, actual_dur


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------

def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int = SAMPLING_RATE) -> None:
    audio = waveform.numpy()
    try:
        import soundfile as sf
        sf.write(str(path), audio, sample_rate)
    except ImportError:
        try:
            import torchaudio
            torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate)
        except ImportError:
            raise ImportError("Install soundfile or torchaudio to save .wav files.")
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode Voxtral TTS voice embeddings to audio waveforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, metavar="DIR",
        help="Path to the Voxtral TTS model directory "
             "(must contain checkpoint shards and optionally params.json/config.json).",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--voice", metavar="PT_FILE",
        help="Path to a single voice embedding .pt file.",
    )
    src.add_argument(
        "--voice-dir", metavar="DIR",
        help="Directory of voice embedding .pt files — decode all.",
    )

    dst = p.add_mutually_exclusive_group()
    dst.add_argument(
        "--output", metavar="WAV_FILE",
        help="Output .wav path (single-file mode).",
    )
    dst.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory for .wav files (batch mode).",
    )

    p.add_argument("--max-iters", type=int, default=10,
                   help="Max coordinate-descent iterations for code reconstruction.")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for codec decoder ('cpu' or 'cuda').",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Decoder weight dtype.",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype  = dtype_map[args.dtype]
    device = args.device
    model_path = Path(args.model)

    # Load decoder weights + embedding table (once, shared across all files)
    tokenizer, emb_table = load_decoder_and_embtable(model_path, dtype=dtype, device=device)

    # ---- Single file ----
    if args.voice:
        voice_path = Path(args.voice)
        waveform, dur = voice_embedding_to_audio(
            voice_path, tokenizer, emb_table, args.max_iters, device
        )
        out = Path(args.output) if args.output else voice_path.with_suffix(".wav")
        save_wav(out, waveform)
        print(f"Duration: {dur:.2f}s  |  Samples: {len(waveform)}  |  Saved: {out}")

    # ---- Batch mode ----
    else:
        voice_dir  = Path(args.voice_dir)
        out_dir    = Path(args.output_dir) if args.output_dir else voice_dir / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        pt_files = sorted(voice_dir.glob("*.pt"))
        if not pt_files:
            logger.error("No .pt files found in %s", voice_dir)
            sys.exit(1)

        print(f"\nDecoding {len(pt_files)} voice embedding(s) → {out_dir}\n")
        for voice_path in pt_files:
            try:
                waveform, dur = voice_embedding_to_audio(
                    voice_path, tokenizer, emb_table, args.max_iters, device
                )
                out = out_dir / voice_path.with_suffix(".wav").name
                save_wav(out, waveform)
                print(f"  {voice_path.name:<35}  {dur:.2f}s  →  {out.name}")
            except Exception as exc:
                logger.error("  Failed %s: %s", voice_path.name, exc, exc_info=True)

        print(f"\nDone — {len(pt_files)} file(s) written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
