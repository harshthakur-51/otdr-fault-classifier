"""Compact statistical features for OTDR traces."""

from __future__ import annotations

import math
import statistics


def extract_features(trace: list[float]) -> list[float]:
    derivatives = [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]
    abs_derivatives = [abs(value) for value in derivatives]
    steepest_up = max(derivatives)
    steepest_down = min(derivatives)
    peak_index = derivatives.index(steepest_up) / len(derivatives)
    fall_index = derivatives.index(steepest_down) / len(derivatives)
    first = statistics.mean(trace[:16])
    last = statistics.mean(trace[-16:])
    reflection_count = sum(1 for value in derivatives if value > 0.055)
    loss_count = sum(1 for value in derivatives if value < -0.055)
    return [
        first - last,
        steepest_up,
        -steepest_down,
        statistics.mean(abs_derivatives),
        statistics.pstdev(trace),
        max(trace) - min(trace),
        peak_index,
        fall_index,
        reflection_count / 3.0,
        loss_count / 3.0,
        math.sqrt(sum(value * value for value in derivatives)),
    ]


def normalize(
    training: list[list[float]], data: list[list[float]]
) -> list[list[float]]:
    means = [statistics.mean(values) for values in zip(*training)]
    scales = [statistics.pstdev(values) or 1.0 for values in zip(*training)]
    return [
        [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]
        for row in data
    ]
