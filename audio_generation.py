# SPDX-License-Identifier: Apache-2.0
# Pure PyTorch / Transformers port of Voxtral TTS audio generation.
# Adapted from vllm-omni — all vLLM dependencies removed.

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedRMSNorm
    rms_norm = FusedRMSNorm
except ImportError:
    try:
        from torch.nn import RMSNorm
        rms_norm = RMSNorm
    except ImportError:
        # PyTorch < 2.4 fallback
        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
                return x / norm * self.weight

        rms_norm = RMSNorm


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

class AudioSpecialTokens(str, Enum):
    """Special tokens predicted by audio codebook heads.

    Audio tokens from the quantizer are offset by ``len(AudioSpecialTokens)``
    so that index 0 and 1 are reserved for these two markers.
    """

    empty_audio = "[EMPTY_AUDIO]"  # id=0 – absence / padding
    end_audio = "[END_AUDIO]"      # id=1 – end of audio sequence

    @staticmethod
    def all_special_tokens() -> list[AudioSpecialTokens]:
        return list(AudioSpecialTokens)

    @staticmethod
    def id(token: AudioSpecialTokens) -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AcousticTransformerArgs:
    input_dim: int
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5


@dataclass
class MultimodalAudioModelArgs:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size] * self.n_acoustic_codebook,
        ]

    def get_codebook_sizes(
        self,
        pad_to_multiple: int | None = 128,
        include_special_tokens: bool = True,
    ) -> list[int]:
        def _round_up(n: int, m: int) -> int:
            return m * ((n + m - 1) // m)

        result: list[int] = []
        for cb_size in self.codebook_sizes:
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up(cb_size, pad_to_multiple)
            result.append(cb_size)
        return result


# ---------------------------------------------------------------------------
# Utility: build dataclass from nested dict
# ---------------------------------------------------------------------------

def from_nested_dict(cls: type, d: dict) -> Any:
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    if not is_dataclass(cls):
        return d
    # get_type_hints() resolves forward-ref strings (created by `from __future__ import annotations`)
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = hints.get(f.name, f.type)
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]
        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats)
        values = _repeat_interleave(values, repeats)
    return keys, values


class FeedForward(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    """Full (non-causal) multi-head attention without positional encoding."""

    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.head_dim: int = args.head_dim
        self.repeats: int = args.n_heads // args.n_kv_heads
        self.layer_id = layer_id

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

    def _native_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scale = query.shape[-1] ** -0.5
        q = (query * scale).transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)).softmax(-1)
        return (attn @ v).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.dim() == 2:
            bsz = 1
            seqlen = x.shape[0]
        else:
            bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xk, xv = repeat_kv(xk, xv, self.repeats)
        output = self._native_attention(xq, xk, xv).view(bsz, seqlen, -1)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow-matching timesteps."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi,j->bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


# ---------------------------------------------------------------------------
# Flow-matching acoustic transformer
# ---------------------------------------------------------------------------

class FlowMatchingAudioTransformer(nn.Module):
    """Predicts per-frame audio codes (37 codebooks) from LLM hidden states.

    Given the last-position hidden state from the LLM backbone, this module:
    1. Greedily predicts the semantic code (codebook 0) via a linear head.
    2. Runs a flow-matching ODE (8 Euler steps, CFG alpha=1.2) to predict the
       36 acoustic codes (codebooks 1-36).

    Input:
        llm_hidden: [B, D] tensor of LLM hidden states.
    Output:
        audio_codes: [B, 37] int64 tensor.
            Values use AudioSpecialTokens offset (0=EMPTY, 1=END,
            2..=actual audio codes).
    """

    def __init__(self, audio_model_args: dict) -> None:
        super().__init__()
        # Support legacy ``codebook_sizes`` string key
        if "codebook_sizes" in audio_model_args:
            audio_model_args = dict(audio_model_args)
            cb_str: str = audio_model_args.pop("codebook_sizes")
            codebook_sizes = [int(c) for c in cb_str.split(",")]
            audio_model_args.update(
                {
                    "semantic_codebook_size": codebook_sizes[0],
                    "acoustic_codebook_size": codebook_sizes[1],
                    "n_acoustic_codebook": len(codebook_sizes) - 1,
                }
            )

        self.model_args: MultimodalAudioModelArgs = from_nested_dict(
            MultimodalAudioModelArgs, audio_model_args
        )
        args = self.model_args.acoustic_transformer_args
        self.acoustic_transformer_args = args

        self.num_non_acoustic_embeddings = 1
        acoustic_cb_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert len(set(acoustic_cb_sizes)) == 1, "All acoustic codebooks must share the same size"
        self.acoustic_embeddings_levels: int = acoustic_cb_sizes[0]
        self.acoustic_embeddings_dim: int = len(acoustic_cb_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        # Flow-matching hyper-parameters (match original)
        self._acoustic_decode_iters = 8
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self._acoustic_decode_iters),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_audio_embeddings_layer(self) -> None:
        args = self.acoustic_transformer_args
        self.time_embedding = TimeEmbedding(args.dim)
        self.input_projection = nn.Linear(self.acoustic_embeddings_dim, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)

    def _init_output_layer(self) -> None:
        args = self.acoustic_transformer_args
        padded_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            args.dim, padded_sizes[0], bias=args.use_biases
        )
        self.acoustic_codebook_output = nn.Linear(
            args.dim, self.model_args.n_acoustic_codebook, bias=False
        )

    def _init_layers(self) -> None:
        args = self.acoustic_transformer_args
        self.layers_ids: list[int] = list(range(args.n_layers))
        self.layers = nn.ModuleDict(
            {str(i): AcousticTransformerBlock(layer_id=i, args=args) for i in self.layers_ids}
        )
        self.norm = rms_norm(args.dim, eps=args.norm_eps)

    # ------------------------------------------------------------------
    # Internal forward helpers
    # ------------------------------------------------------------------

    def _forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for i in self.layers_ids:
            h = self.layers[str(i)](h)
        return h

    def _predict_velocity(
        self,
        x_t: torch.Tensor,       # [2B, C]
        llm_output: torch.Tensor, # [2B, D]
        t_emb: torch.Tensor,     # [2B, D]
    ) -> torch.Tensor:
        x_t = x_t.to(llm_output.dtype)
        t_proj = self.time_projection(t_emb)
        llm_proj = self.llm_projection(llm_output)
        # Sequence: [x_t, t, llm] — shape [2B, 3, dim]
        inputs = torch.cat(
            [
                self.input_projection(x_t.unsqueeze(1)),
                t_proj.unsqueeze(1),
                llm_proj.unsqueeze(1),
            ],
            dim=1,
        )
        out = self.norm(self._forward_attention_layers(inputs))
        # Predict velocity from the x_t token (position 0)
        return self.acoustic_codebook_output(out[:, 0, :])

    def decode_one_frame(
        self,
        semantic_code: torch.Tensor,  # [B]
        llm_hidden: torch.Tensor,     # [B, D]
    ) -> torch.Tensor:
        """Flow-matching ODE: random noise → acoustic codes for one frame.

        Returns [B, n_acoustic_codebook] int64 tensor with AudioSpecialTokens offset.
        """
        B = semantic_code.shape[0]
        should_decode = semantic_code != self._end_audio_token_id

        x = self._noise_scale * torch.randn(
            B, self.model_args.n_acoustic_codebook,
            dtype=llm_hidden.dtype, device=llm_hidden.device,
        )
        timesteps = self._timesteps.to(dtype=llm_hidden.dtype)
        llm_zero = torch.zeros_like(llm_hidden)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = self.time_embedding(t.view(-1, 1).expand(B, 1)).to(llm_hidden.dtype)

            # Conditional + unconditional in a single 2B forward pass (CFG)
            v_all = self._predict_velocity(
                x_t=torch.cat([x, x], dim=0),
                llm_output=torch.cat([llm_hidden, llm_zero], dim=0),
                t_emb=torch.cat([t_emb, t_emb], dim=0),
            )
            v_cond, v_uncond = v_all[:B], v_all[B:]
            v = self._cfg_alpha * v_cond + (1 - self._cfg_alpha) * v_uncond
            x = x + v * dt

        # Quantize continuous values to integer codes
        x = torch.clamp(x, -1.0, 1.0)
        codes = (((x + 1) / 2) * (self.acoustic_embeddings_levels - 1)).round().long()
        codes[~should_decode] = self._empty_audio_token_id
        return codes + len(AudioSpecialTokens)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llm_hidden: [B, D] last-layer hidden states from the LLM backbone.

        Returns:
            audio_codes: [B, 37] int64 tensor.
                - Column 0: semantic code (0=EMPTY, 1=END, 2+ = code + 2)
                - Columns 1-36: acoustic codes (same offset convention)
        """
        sem_logit = self.semantic_codebook_output(llm_hidden).float()
        sem_logit[:, self._empty_audio_token_id] = float("-inf")
        sem_logit[:, len(AudioSpecialTokens) + self.model_args.semantic_codebook_size :] = float("-inf")
        semantic_code = sem_logit.argmax(dim=-1, keepdim=True)  # [B, 1]

        acoustic_codes = self.decode_one_frame(semantic_code.squeeze(1), llm_hidden)  # [B, 36]
        return torch.cat([semantic_code, acoustic_codes], dim=1)  # [B, 37]
