from __future__ import annotations

import math

import torch


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    # FlashVSR 1.1/Wan computes this in float64 and casts back to the
    # timestep dtype. Keeping that detail matters for BF16 checkpoint parity.
    frequencies = torch.outer(
        position.to(torch.float64).flatten(),
        torch.pow(
            10000.0,
            -torch.arange(dim // 2, device=position.device, dtype=torch.float64)
            / (dim // 2),
        ),
    )
    return torch.cat([frequencies.cos(), frequencies.sin()], dim=-1).to(position.dtype)


def _axis_pair_counts(num_pairs: int) -> tuple[int, int, int]:
    base = num_pairs // 3
    return num_pairs - 2 * base, base, base


def rope_angles_3d(
    shape: tuple[int, int, int],
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
    temporal_offset: int = 0,
) -> torch.Tensor:
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frames, height, width = shape
    pair_counts = _axis_pair_counts(head_dim // 2)
    axes = (
        torch.arange(
            temporal_offset, temporal_offset + frames, device=device
        ).view(frames, 1, 1).expand(frames, height, width),
        torch.arange(height, device=device).view(1, height, 1).expand(frames, height, width),
        torch.arange(width, device=device).view(1, 1, width).expand(frames, height, width),
    )
    chunks = []
    for positions, pairs in zip(axes, pair_counts):
        inverse = theta ** (-torch.arange(pairs, device=device).float() / max(pairs, 1))
        chunks.append(positions.reshape(-1, 1).float() * inverse)
    return torch.cat(chunks, dim=-1)


_ROPE_TRIG_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_rope_cos_sin(
    shape: tuple[int, int, int],
    head_dim: int,
    device: torch.device,
    temporal_offset: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_key = (device.type, device.index if device.index is not None else 0)
    cache_key = (shape, head_dim, device_key, temporal_offset, dtype)
    entry = _ROPE_TRIG_CACHE.get(cache_key)
    if entry is None:
        angles = rope_angles_3d(
            shape, head_dim, device, temporal_offset=temporal_offset
        )
        cos = angles.cos().view(1, 1, angles.shape[0], -1).to(dtype=dtype)
        sin = angles.sin().view(1, 1, angles.shape[0], -1).to(dtype=dtype)
        entry = (cos, sin)
        _ROPE_TRIG_CACHE[cache_key] = entry
    return entry


def apply_rope_3d(
    x: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    temporal_offset: int = 0,
) -> torch.Tensor:
    cos, sin = _get_rope_cos_sin(
        shape, x.shape[-1], x.device, temporal_offset, x.dtype
    )
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    first, second = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.empty_like(x_pairs)
    rotated[..., 0] = first * cos - second * sin
    rotated[..., 1] = second * cos + first * sin
    return rotated.flatten(-2)
