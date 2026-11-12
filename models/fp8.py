"""FP8 quantization utilities for weights and KV cache on Ada Lovelace and newer GPUs."""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


FP8_MAX = 448.0


class FP8Linear(nn.Module):
    """Linear layer storing weights in torch.float8_e4m3fn with column-major cuBLASLt layout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("weight_fp8", weight_fp8)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, orig: nn.Linear) -> "FP8Linear":
        w = orig.weight.data
        scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12).float()
        # cuBLASLt expects column-major weight: [K, N] with stride(0)==1, stride(1)==K
        w_col = (w / scale.to(w.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).t()
        bias = orig.bias.data.clone().to(torch.bfloat16) if orig.bias is not None else None
        return cls(orig.in_features, orig.out_features, w_col, scale, bias)

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse=recurse)
        if hasattr(self, "weight_fp8") and self.weight_fp8 is not None:
            if self.weight_fp8.dtype != torch.float8_e4m3fn:
                self.weight_fp8 = self.weight_fp8.to(torch.float8_e4m3fn)
        if hasattr(self, "weight_scale") and self.weight_scale is not None:
            if self.weight_scale.dtype != torch.float32:
                self.weight_scale = self.weight_scale.to(torch.float32)
        return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        M = x_2d.shape[0]
        # Fast path with native Ada cuBLASLt torch._scaled_mm when conditions are met
        if (
            x.is_cuda
            and M % 16 == 0
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        ):
            scale_x = (x_2d.abs().max() / FP8_MAX).clamp(min=1e-12).float()
            x_fp8 = (x_2d / scale_x.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
            out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8,
                scale_a=scale_x,
                scale_b=self.weight_scale,
                out_dtype=x.dtype,
            )
        else:
            # Fallback path: dequantize weight on the fly
            w = self.weight_fp8.t().to(x.dtype) * self.weight_scale.to(x.dtype)
            out = F.linear(x_2d, w)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)


def quantize_model_fp8(module: nn.Module, *, min_features: int = 512, prefix: str = "", compile_layers: bool = False) -> int:
    """Recursively convert large nn.Linear layers to FP8Linear."""
    converted = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if child.in_features >= min_features and child.out_features >= min_features:
                fp8_mod = FP8Linear.from_linear(child).to(device=child.weight.device)
                if compile_layers:
                    try:
                        fp8_mod = torch.compile(fp8_mod, mode="default")
                    except Exception:
                        pass
                setattr(module, name, fp8_mod)
                converted += 1
        else:
            converted += quantize_model_fp8(child, min_features=min_features, prefix=full_name, compile_layers=compile_layers)
    return converted


def quantize_kv_cache(k: torch.Tensor, v: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Quantize K and V tensors to FP8 with per-tensor scales."""
    scale_k = (k.abs().max() / FP8_MAX).clamp(min=1e-12).float()
    k_fp8 = (k / scale_k.to(k.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    scale_v = (v.abs().max() / FP8_MAX).clamp(min=1e-12).float()
    v_fp8 = (v / scale_v.to(v.dtype)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return ((k_fp8, scale_k), (v_fp8, scale_v))


def dequantize_kv_cache(cache_k: Any, cache_v: Any, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize FP8 K and V tensors back to target tensor dtype and device."""
    if isinstance(cache_k, (tuple, list)):
        k_val, scale_k = cache_k
        k_out = (k_val.to(target.dtype) * scale_k.to(target.dtype)).to(target.device)
    else:
        k_out = cache_k.to(target)
    if isinstance(cache_v, (tuple, list)):
        v_val, scale_v = cache_v
        v_out = (v_val.to(target.dtype) * scale_v.to(target.dtype)).to(target.device)
    else:
        v_out = cache_v.to(target)
    return k_out, v_out


__all__ = [
    "FP8Linear",
    "quantize_model_fp8",
    "quantize_kv_cache",
    "dequantize_kv_cache",
]

