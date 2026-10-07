"""PCM helpers: resampling, chunking, durations, comfort sounds."""

from __future__ import annotations

import numpy as np


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(x) * dst_rate / src_rate)
    if n_out <= 0:
        return b""
    t_in = np.linspace(0, 1, len(x), endpoint=False)
    t_out = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.int16).tobytes()


def duration_s(pcm: bytes, rate: int) -> float:
    return len(pcm) / 2 / rate


def chunk(pcm: bytes, size: int) -> list[bytes]:
    """Split into chunks of `size` bytes; pad the last to a multiple of 320 (Exotel requirement)."""
    out = [pcm[i: i + size] for i in range(0, len(pcm), size)]
    if out and len(out[-1]) % 320:
        out[-1] += b"\x00" * (320 - len(out[-1]) % 320)
    return out


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def keyboard_clicks(rate: int, seconds: float = 1.5, seed: int = 3) -> bytes:
    """Soft synthetic typing sound for slow lookups (stand-in for the course repo's keyboard.mp3)."""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    out = np.zeros(n, dtype=np.float32)
    t = 0
    while t < n:
        t += int(rate * rng.uniform(0.07, 0.22))
        L = int(rate * 0.012)
        if t + L < n:
            out[t: t + L] += rng.normal(0, 900, L) * np.exp(-np.linspace(0, 6, L))
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
