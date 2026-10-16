from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Window:
    window_index: int
    start_ms: float
    end_ms: float

    @property
    def partial(self):
        return self.end_ms - self.start_ms < 10000

    @property
    def targets(self):
        return [self.start_ms + (i+0.5)*(self.end_ms-self.start_ms)/12 for i in range(12)]


def generate_windows(duration_ms: float) -> list[Window]:
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("SOURCE_INVALID: duration must be finite and positive")
    return [Window(k, k*5000, min(k*5000+10000, duration_ms))
            for k in range(max(1, math.floor(duration_ms/5000)))]


def nearest_samples(frames, window: Window):
    """Keep all 12 positions; equal distances select the earlier source PTS."""
    import bisect

    candidates = sorted((f for f in frames if window.start_ms <= f[0].timestamp_ms < window.end_ms),
                        key=lambda f: (f[0].timestamp_ms, f[0].pts))
    if not candidates:
        raise ValueError("DECODE_EMPTY_WINDOW: no source frames inside window")
    timestamps = [f[0].timestamp_ms for f in candidates]
    selected = []
    for target in window.targets:
        pos = bisect.bisect_left(timestamps, target)
        choices = [candidates[i] for i in {max(0, pos-1), min(pos, len(candidates)-1)}]
        selected.append(min(choices, key=lambda f: (abs(f[0].timestamp_ms-target), f[0].timestamp_ms, f[0].pts)))
    return selected


def pts_ms(pts: int, origin: int, numerator: int, denominator: int) -> float:
    return float(Fraction((pts-origin)*numerator*1000, denominator))
