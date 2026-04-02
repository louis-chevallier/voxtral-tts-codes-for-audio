# SPDX-License-Identifier: Apache-2.0
# Pure PyTorch / Transformers port of Voxtral TTS audio tokenizer (codec).
# Adapted from vllm-omni — all vLLM dependencies removed.

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False

try:
    from apex.normalization import FusedRMSNorm
    rms_norm = FusedRMSNorm
except ImportError:
    try:
        from torch.nn import RMSNorm
        rms_norm = RMSNorm
    except ImportError:
        # PyTorch < 2.4 fallback
        class RMSNorm(nn.Module):  # type: ignore[no-redef]
            def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
                return x / norm * self.weight

        rms_norm = RMSNorm

from audio_generation import (
    AudioSpecialTokens,
    FeedForward,
    MultimodalAudioModelArgs,
    from_nested_dict,
)

logger = logging.getLogger(__name__)

if not HAS_FLASH_ATTN:
    logger.warning(
        "flash_attn not installed — falling back to PyTorch SDPA for codec attention. "
        "Install flash-attn for better performance."
    )

weight_norm = torch.nn.utils.parametrizations.weight_norm

CODEC_NORM_EPS = 1e-2


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AudioTokenizerArgs:
    # Audio
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7

    # Quantizer
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36

    # Architecture (general)
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 1e-2
    layer_scale: bool = True
    layer_scale_init: float | None = None

    # Encoder
    encoder_transformer_lengths_str: str = "2,2,2,2"
    encoder_convs_kernels_str: str = "4,4,4,3"
    encoder_convs_strides_str: str = "2,2,2,1"

    # Decoder
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"

    def __post_init__(self) -> None:
        assert len(self.encoder_transformer_lengths) == len(self.encoder_convs_kernels) == len(self.encoder_convs_strides)
        assert len(self.decoder_transformer_lengths) == len(self.decoder_convs_kernels) == len(self.decoder_convs_strides)

    def _str2list(self, s: str) -> tuple[int, ...]:
        return tuple(int(i) for i in s.split(","))

    @property
    def encoder_transformer_lengths(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_transformer_lengths_str)

    @property
    def encoder_convs_kernels(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_convs_kernels_str)

    @property
    def encoder_convs_strides(self) -> tuple[int, ...]:
        return self._str2list(self.encoder_convs_strides_str)

    @property
    def decoder_transformer_lengths(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_transformer_lengths_str)

    @property
    def decoder_convs_kernels(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_convs_kernels_str)

    @property
    def decoder_convs_strides(self) -> tuple[int, ...]:
        return self._str2list(self.decoder_convs_strides_str)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / (
            self.pretransform_patch_size * math.prod(self.encoder_convs_strides)
        )


# ---------------------------------------------------------------------------
# Quantizers
# ---------------------------------------------------------------------------

class SemanticCodebook(nn.Module):
    """Euclidean-distance codebook for semantic quantization."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon: float = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            emb = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self._embedding: torch.Tensor
            self.register_buffer("_embedding", emb, persistent=False)
            return emb
        return self._embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, T] → codes: [B, 1, T]"""
        B, D, T = x.shape
        flat = rearrange(x, "b d t -> (b t) d")
        distances = torch.cdist(flat, self.embedding.to(flat.device), p=2)
        codes = distances.argmin(dim=-1).view(B, 1, T)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [B, 1, T] → [B, D, T]"""
        codes = codes.squeeze(1)  # [B, T]
        quantized = F.embedding(codes, self.embedding.to(codes.device))
        return rearrange(quantized, "b t d -> b d t")

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_sizes(self) -> list[int]:
        return [self.cluster_usage.shape[0]]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization for acoustic codebooks."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.n_levels = codebook_size
        self.num_codebooks = codebook_dim

    @property
    def codebook_sizes(self) -> list[int]:
        return [self.n_levels] * self.num_codebooks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, T] → codes: [B, D, T]"""
        x = torch.tanh(x)
        levels = torch.ones_like(x) * self.n_levels
        scaled = ((x + 1) / 2) * (levels - 1)
        return scaled.round().long()

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: [B, D, T] → [B, D, T] float"""
        return ((codes.to(dtype) * 2 / (self.n_levels - 1)) - 1)


class MistralAudioCodebook(nn.Module):
    """Combined semantic + acoustic quantizer."""

    def __init__(self, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self.semantic_codebook = SemanticCodebook(args.semantic_codebook_size, args.semantic_dim)
        self.acoustic_codebook = AcousticCodebook(args.acoustic_codebook_size, args.acoustic_dim)
        self.semantic_dim = args.semantic_dim
        self.acoustic_dim = args.acoustic_dim
        self.total_dim = self.semantic_dim + self.acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return self.semantic_codebook.num_codebooks + self.acoustic_codebook.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        return self.semantic_codebook.codebook_sizes + self.acoustic_codebook.codebook_sizes

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, T] → codes: [B, K, T] where K = num_codebooks"""
        semantic_codes = self.semantic_codebook.encode(x[:, : self.semantic_dim, :])
        acoustic_codes = self.acoustic_codebook.encode(x[:, self.semantic_dim :, :])
        return torch.cat([semantic_codes, acoustic_codes], dim=1)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: [B, K, T] → [B, D, T]"""
        n_sem = self.semantic_codebook.num_codebooks
        sem_emb = self.semantic_codebook.decode(codes[:, :n_sem, :]).to(dtype)
        aco_emb = self.acoustic_codebook.decode(codes[:, n_sem:, :], dtype=dtype)
        return torch.cat([sem_emb, aco_emb], dim=1)


# ---------------------------------------------------------------------------
# Multi-vocabulary embedding table for audio token lookup
# ---------------------------------------------------------------------------

class MultiVocabEmbeddings(nn.Module):
    """Per-codebook embedding table.  Each codebook's tokens are stored in a
    contiguous block of the shared embedding matrix."""

    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        import numpy as np  # local import to avoid top-level dep
        model_args: MultimodalAudioModelArgs = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = [c for c in model_args.get_codebook_sizes(pad_to_multiple=None)]
        self.offsets = torch.from_numpy(np.cumsum([0] + self.codebook_sizes[:-1]))
        total = sum(self.codebook_sizes)
        padded = 128 * ((total + 127) // 128)
        self.embeddings = nn.Embedding(padded, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, C, L] → [B, C, L, D]"""
        self.offsets = self.offsets.to(input_ids.device)
        shifted = input_ids + self.offsets[None, :, None]
        return self.embeddings(shifted)


# ---------------------------------------------------------------------------
# Causal convolutions
# ---------------------------------------------------------------------------

def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """F.pad wrapper that handles reflect mode on short inputs."""
    length = x.shape[-1]
    pad_left, pad_right = paddings
    assert pad_left >= 0 and pad_right >= 0
    if mode == "reflect":
        max_pad = max(pad_left, pad_right)
        extra = 0
        if length <= max_pad:
            extra = max_pad - length + 1
            x = F.pad(x, (0, extra))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.pad_mode = pad_mode
        self._stride = self.conv.stride[0]
        self._effective_kernel_size = (kernel_size - 1) * self.conv.dilation[0] + 1
        self._padding_total = self._effective_kernel_size - self._stride
        self.stride = self.conv.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (
            (math.ceil(n_frames) - 1) * self._stride
            + (self._effective_kernel_size - self._padding_total)
        )
        extra_padding = target_length - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        trim_ratio: float = 1.0,
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, groups=groups, bias=use_bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


# ---------------------------------------------------------------------------
# Codec transformer (with ALiBi + sliding window + optional flash-attn)
# ---------------------------------------------------------------------------

def prepare_for_attention(x: torch.Tensor, time_last: bool = True) -> torch.Tensor:
    if time_last:
        return rearrange(x, "b d t -> (b t) d")
    return rearrange(x, "b t d -> (b t) d")


class Attention(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.repeats = args.n_heads // args.n_kv_heads
        self.layer_id = layer_id
        self.sliding_window = args.attn_sliding_window_size

        def _alibi_slopes(n: int) -> torch.Tensor:
            def _pow2(n: int) -> torch.Tensor:
                r = 2.0 ** (-8.0 / n)
                return torch.tensor([r ** i for i in range(n)], dtype=torch.float32)

            if math.log2(n).is_integer():
                return _pow2(n)
            m = 2 ** math.floor(math.log2(n))
            return torch.cat([_pow2(m), _pow2(2 * m)[::2][: n - m]])

        self.register_buffer("alibi_slopes", _alibi_slopes(self.n_local_heads), persistent=False)

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        if args.qk_norm:
            self.q_norm = rms_norm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = rms_norm(args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps)

    def _native_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H, D = xq.shape
        Hkv = xk.shape[2]

        q = xq.transpose(1, 2)  # [B, H, S, D]
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if H != Hkv:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)

        positions = torch.arange(S, device=xq.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]

        slopes = self.alibi_slopes.to(dtype=xq.dtype, device=xq.device)
        attn_bias = slopes.view(H, 1, 1) * rel_pos.unsqueeze(0).to(xq.dtype)

        if self.args.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        win_left = self.sliding_window
        win_right = 0 if self.args.causal else self.sliding_window
        outside = (rel_pos < -win_left) | (rel_pos > win_right)
        attn_bias = attn_bias.masked_fill(outside.unsqueeze(0), float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(0))
        return out.transpose(1, 2)  # [B, S, H, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.args.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        if HAS_FLASH_ATTN:
            slopes = self.alibi_slopes.to(torch.float32)
            output = flash_attn_func(
                xq, xk, xv,
                causal=self.args.causal,
                window_size=(self.sliding_window, 0 if self.args.causal else self.sliding_window),
                alibi_slopes=slopes,
            )
        else:
            output = self._native_attention(xq, xk, xv)

        output = output.view(bsz, seqlen, self.n_local_heads * self.args.head_dim)
        return self.wo(output).squeeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AudioTokenizerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.attention = Attention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.post_attention_norm: nn.Module | None = None
        self.post_ffn_norm: nn.Module | None = None
        self.args = args

        self.layer_scale = args.layer_scale
        if self.layer_scale:
            if args.layer_scale_init is None:
                if layer_id < 18:
                    init_val = 0.1
                elif layer_id <= 24:
                    init_val = 1e-5
                else:
                    init_val = 1e-6
            else:
                init_val = args.layer_scale_init
            self.attention_scale = nn.Parameter(torch.full((args.dim,), init_val))
            self.ffn_scale = nn.Parameter(torch.full((args.dim,), init_val))

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        if self.post_attention_norm is not None:
            r = self.post_attention_norm(r)
        if self.layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.post_ffn_norm is not None:
            r = self.post_ffn_norm(r)
        if self.layer_scale:
            r = self.ffn_scale * r
        return h + r


class Transformer(nn.Module):
    def __init__(self, args: AudioTokenizerArgs, n_layers: int) -> None:
        super().__init__()
        self.layers_ids = list(range(n_layers))
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(layer_id=i, args=args) for i in self.layers_ids}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in self.layers_ids:
            x = self.layers[str(i)](x)
        return x


# ---------------------------------------------------------------------------
# Audio tokenizer (encoder + decoder + quantizer + audio token embedding)
# ---------------------------------------------------------------------------

class VoxtralTTSAudioTokenizer(nn.Module):
    """Audio codec for Voxtral TTS.

    Contains:
    - Encoder (optional — may be absent in open-source checkpoints)
    - Quantizer (semantic + acoustic codebooks)
    - Decoder (always present)
    - ``audio_token_embedding``: multi-codebook embedding table used to
      convert audio token IDs from the LLM into dense vectors.

    For TTS inference the decoder and embedding table are the parts you need.
    The encoder is only required for voice-cloning from raw waveforms.
    """

    def __init__(
        self,
        codec_args: dict,
        audio_model_args: dict,
        text_hidden_size: int,
    ) -> None:
        super().__init__()
        args: AudioTokenizerArgs = from_nested_dict(AudioTokenizerArgs, codec_args)
        self.args = args

        if not args.causal:
            raise NotImplementedError("Only causal mode is implemented")

        self.patch_size = args.pretransform_patch_size
        self.latent_dim = args.semantic_dim + args.acoustic_dim

        # ---- Input patch projection ----
        self.input_proj = CausalConv1d(
            args.pretransform_patch_size * args.channels,
            args.dim,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        # ---- Encoder ----
        encoder_blocks: list[nn.Module] = []
        cur_window_size = args.attn_sliding_window_size

        for idx, n_layers in enumerate(args.encoder_transformer_lengths):
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            encoder_blocks.append(Transformer(args=layer_args, n_layers=n_layers))

            is_last = idx == len(args.encoder_transformer_lengths) - 1
            proj_out_dim = self.latent_dim if is_last else args.dim
            k = args.encoder_convs_kernels[idx]
            s = args.encoder_convs_strides[idx]
            if k != 1 or s != 1 or is_last:
                encoder_blocks.append(
                    CausalConv1d(args.dim, proj_out_dim, kernel_size=k, stride=s, pad_mode="replicate", use_bias=False)
                )
                if args.half_attn_window_upon_downsampling and s > 1:
                    assert s == 2
                    cur_window_size = cur_window_size // 2
                    assert cur_window_size >= 2

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # ---- Audio token embedding (for LLM → embedding conversion) ----
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_model_args,
            embedding_dim=text_hidden_size,
        )

        # ---- Decoder ----
        decoder_blocks: list[nn.Module] = []

        # First decoder conv: latent_dim → dim
        k0 = args.decoder_convs_kernels[0]
        s0 = args.decoder_convs_strides[0]
        decoder_blocks.append(
            CausalConv1d(self.latent_dim, args.dim, kernel_size=k0, stride=s0, pad_mode="replicate", use_bias=False)
        )
        if args.half_attn_window_upon_downsampling and s0 > 1:
            assert s0 == 2
            cur_window_size = cur_window_size * 2

        for idx, n_layers in enumerate(args.decoder_transformer_lengths):
            layer_args = deepcopy(args)
            layer_args.attn_sliding_window_size = cur_window_size
            decoder_blocks.append(Transformer(args=layer_args, n_layers=n_layers))

            # Upsample after each block (except the last)
            if idx + 1 < len(args.decoder_transformer_lengths):
                k = args.decoder_convs_kernels[idx + 1]
                s = args.decoder_convs_strides[idx + 1]
                if k != 1 or s != 1:
                    decoder_blocks.append(
                        CausalConvTranspose1d(args.dim, args.dim, kernel_size=k, stride=s, use_bias=False)
                    )
                    if args.half_attn_window_upon_downsampling and s > 1:
                        assert s == 2
                        cur_window_size = cur_window_size * 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        # ---- Quantizer ----
        self.quantizer = MistralAudioCodebook(args)

        # ---- Output patch projection ----
        self.output_proj = CausalConv1d(
            args.dim,
            args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            use_weight_norm=args.conv_weight_norm,
            use_bias=False,
        )

        scale_factor = math.prod(args.encoder_convs_strides)
        assert scale_factor == math.prod(args.decoder_convs_strides)
        self._frame_rate = args.sampling_rate / (self.patch_size * scale_factor)
        self._sampling_rate = args.sampling_rate
        self._channels = args.channels

        # Track whether encoder weights were loaded
        self._encoder_weight_prefixes = ("input_proj.", "encoder_blocks.")
        self._encoder_loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def downsample_factor(self) -> int:
        assert self._sampling_rate % self._frame_rate == 0
        return int(self._sampling_rate / self._frame_rate)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def codebook_sizes(self) -> list[int]:
        return self.quantizer.codebook_sizes

    # ------------------------------------------------------------------
    # Weight loading helper (called by VoxtralTTS.load_weights)
    # ------------------------------------------------------------------

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        """Load a single weight tensor by name (post-remapping)."""
        if name == "quantizer.semantic_codebook.cluster_usage":
            self.quantizer.semantic_codebook.cluster_usage.copy_(tensor)
            return
        if name == "quantizer.semantic_codebook.embedding_sum":
            self.quantizer.semantic_codebook.embedding_sum.copy_(tensor)
            # Invalidate cached embedding
            self.quantizer.semantic_codebook._embedding = None
            return

        params = dict(self.named_parameters())
        if name not in params:
            logger.warning("VoxtralTTSAudioTokenizer: weight '%s' not found — skipping", name)
            return
        params[name].data.copy_(tensor)
        if any(name.startswith(p) for p in self._encoder_weight_prefixes):
            self._encoder_loaded = True

    # ------------------------------------------------------------------
    # Encoder path
    # ------------------------------------------------------------------

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T_samples] → [B, latent_dim, T_frames]"""
        assert x.dim() == 3
        emb = rearrange(x, "b c (t h) -> b (c h) t", h=self.patch_size)
        emb = self.input_proj(emb)
        emb = rearrange(emb, "b d t -> b t d").contiguous()

        for block in self.encoder_blocks:
            if isinstance(block, CausalConv1d):
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                bsz = emb.shape[0]
                emb = prepare_for_attention(emb, time_last=False)  # [(B*T), D]
                emb = block(emb)
                emb = rearrange(emb, "(b t) d -> b t d", b=bsz)

        return rearrange(emb, "b t d -> b d t")

    def _tokenize_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Encode waveform to discrete codes.

        Args:
            x: [B, C, T_samples]

        Returns:
            codes: [B, K, T_frames]
        """
        if x.shape[-1] % self.patch_size != 0:
            pad_len = self.patch_size - (x.shape[-1] % self.patch_size)
            x = F.pad(x, (0, pad_len))

        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(dtype=torch.bfloat16, device_type=device_type):
            emb = self._forward_encoder(x)
        return self.quantizer.encode(emb)

    def encode_waveforms(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode raw waveforms to audio token embeddings (for LLM input).

        Requires encoder weights (may be absent in open-source checkpoints).

        Args:
            waveforms: list of 1-D float tensors in [-1, 1] at ``sampling_rate``.

        Returns:
            List of [T, hidden_size] tensors (one per waveform).
        """
        if not self._encoder_loaded:
            raise RuntimeError(
                "encode_waveforms() requires encoder weights, which are absent "
                "in the open-source Voxtral TTS checkpoint. "
                "Use a pre-computed voice embedding or audio token codes instead."
            )
        audio_codes_list: list[torch.Tensor] = []
        for waveform in waveforms:
            assert waveform.dim() == 1
            codes = self._tokenize_audio(waveform.unsqueeze(0).unsqueeze(0))  # [1, K, T]
            # Offset codes to skip AudioSpecialTokens
            codes = codes + len(AudioSpecialTokens.all_special_tokens())
            # Append end-of-audio token (code 0 of first codebook = 1)
            B, K, _ = codes.shape
            eoa = torch.zeros(B, K, 1, dtype=codes.dtype, device=codes.device)
            eoa[:, 0, 0] = 1  # END_AUDIO in first codebook
            codes = torch.cat([codes, eoa], dim=-1)
            audio_codes_list.append(codes)
        return self.encode_tokens(audio_codes_list)

    def encode_tokens(self, audio_codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Convert pre-quantized audio codes to LLM embeddings.

        Args:
            audio_codes: list of [1, K, T] int tensors.

        Returns:
            List of [T, hidden_size] tensors.
        """
        embeddings: list[torch.Tensor] = []
        for codes in audio_codes:
            # codes: [1, K, T] → embed → [1, K, T, D] → sum over K → [1, T, D] → [T, D]
            emb = self.audio_token_embedding(codes)  # [1, K, T, D]
            emb = emb.sum(dim=1)                      # [1, T, D]
            emb = emb.squeeze(0)                      # [T, D]
            embeddings.append(emb)
        return embeddings

    # ------------------------------------------------------------------
    # Decoder path
    # ------------------------------------------------------------------

    def _forward_decoder(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: [B, latent_dim, T] → [B, C, T_samples]"""
        emb = rearrange(emb, "b d t -> b t d").contiguous()

        for block in self.decoder_blocks:
            if isinstance(block, (CausalConvTranspose1d, CausalConv1d)):
                emb = rearrange(emb, "b t d -> b d t")
                emb = block(emb)
                emb = rearrange(emb, "b d t -> b t d")
            else:
                emb = block(emb)  # Transformer: [B, T, D]

        emb = rearrange(emb, "b t d -> b d t")
        emb = self.output_proj(emb)
        return rearrange(emb, "b (c h) t -> b c (t h)", h=self.patch_size)

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode quantized codes to waveform.

        Args:
            codes: [B, K, T] int tensor.

        Returns:
            out: [B, C, T_samples] float tensor.
        """
        emb = self.quantizer.decode(codes, dtype)
        return self._forward_decoder(emb)

    def decode_helper_batch_async(
        self, codes_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Batch-decode a list of per-request code tensors to waveforms.

        Each element of ``codes_list`` is a [T_i, K] tensor where:
        - T_i varies per request
        - K = num_codebooks (e.g. 37)
        - codes[:, 0] == 1 marks the END_AUDIO frame
        - All codes use the AudioSpecialTokens offset (real codes start at 2)

        Returns:
            List of 1-D float tensors of reconstructed audio samples.
        """
        chunk_size = 375  # max frames per decode chunk

        # Pre-process: find EOA, strip offset, trim
        processed: list[torch.Tensor] = []
        for codes in codes_list:
            eoa_mask = codes[:, 0] == 1  # END_AUDIO id = 1
            eoa_indices = eoa_mask.nonzero(as_tuple=False)
            cut = int(eoa_indices[0].item()) if len(eoa_indices) > 0 else len(codes)
            processed.append(codes[:cut] - 2)  # strip AudioSpecialTokens offset

        # Handle empty results
        results: list[torch.Tensor | None] = [None] * len(processed)
        non_empty: list[tuple[int, torch.Tensor]] = []
        for idx, tokens in enumerate(processed):
            if len(tokens) == 0:
                results[idx] = torch.tensor([], dtype=torch.float32)
            else:
                non_empty.append((idx, tokens))

        if not non_empty:
            return results  # type: ignore[return-value]

        # Split into chunks
        all_chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []
        chunk_map: list[tuple[int, list[int]]] = []

        for orig_idx, tokens in non_empty:
            req_chunk_indices: list[int] = []
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                req_chunk_indices.append(len(all_chunks))
                chunk_lengths.append(len(chunk))
                all_chunks.append(chunk)
            chunk_map.append((orig_idx, req_chunk_indices))

        # Pad to max chunk length, batch decode
        max_len = max(chunk_lengths)
        K = all_chunks[0].shape[1]
        device = all_chunks[0].device
        padded = torch.zeros(len(all_chunks), max_len, K, dtype=all_chunks[0].dtype, device=device)
        for i, chunk in enumerate(all_chunks):
            padded[i, : len(chunk)] = chunk

        # [B, T, K] → transpose → [B, K, T] for decode()
        model_dtype = next(self.parameters()).dtype
        audio_values = self.decode(padded.transpose(1, 2), dtype=model_dtype)  # [B, 1, T_out]
        audio_values = audio_values.detach().cpu().float().squeeze(1)              # [B, T_out]

        # Trim padding and reassemble per request
        for orig_idx, chunk_indices in chunk_map:
            parts: list[torch.Tensor] = []
            for ci in chunk_indices:
                n_samples = chunk_lengths[ci] * self.downsample_factor
                parts.append(audio_values[ci, :n_samples])
            results[orig_idx] = torch.cat(parts) if len(parts) > 1 else parts[0]

        return results  # type: ignore[return-value]
