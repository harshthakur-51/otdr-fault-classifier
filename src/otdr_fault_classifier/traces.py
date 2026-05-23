"""Synthetic OTDR trace generation for a public demonstration dataset."""

from __future__ import annotations

import math
import random

FAULT_CLASSES = (
    "normal",
    "connector_loss",
    "splice_loss",
    "bend",
    "reflection",
    "break",
    "dirty_connector",
    "multiple_events",
)


def _pulse(index: int, center: int, amplitude: float, width: float = 1.8) -> float:
    return amplitude * math.exp(-((index - center) ** 2) / (2.0 * width**2))


def generate_trace(label: str, rng: random.Random, points: int = 128) -> list[float]:
    """Return a noisy OTDR-like attenuation trace for one fault class."""
    if label not in FAULT_CLASSES:
        raise ValueError(f"Unknown fault class: {label}")

    event = rng.randint(40, 85)
    trace = [1.0 - 0.0018 * i + rng.gauss(0.0, 0.0045) for i in range(points)]

    for i in range(points):
        if label == "connector_loss" and i >= event:
            trace[i] -= 0.065
        elif label == "splice_loss" and i >= event:
            trace[i] -= 0.15
        elif label == "bend":
            trace[i] -= 0.12 * math.exp(-((i - event) ** 2) / (2.0 * 10.0**2))
        elif label == "reflection":
            trace[i] += _pulse(i, event, 0.25)
        elif label == "break":
            trace[i] += _pulse(i, event, 0.30)
            if i >= event + 3:
                trace[i] -= 0.52
        elif label == "dirty_connector":
            trace[i] += _pulse(i, event, 0.18) + _pulse(i, event + 6, 0.10)
            if i >= event + 2:
                trace[i] -= 0.08
        elif label == "multiple_events":
            trace[i] += _pulse(i, event, 0.18) + _pulse(i, event + 26, 0.25)
            if i >= event + 4:
                trace[i] -= 0.10
            if i >= event + 29:
                trace[i] -= 0.22

    return trace


def generate_dataset(
    samples_per_class: int = 80, seed: int = 7
) -> list[tuple[list[float], str]]:
    rng = random.Random(seed)
    rows = [
        (generate_trace(label, rng), label)
        for label in FAULT_CLASSES
        for _ in range(samples_per_class)
    ]
    rng.shuffle(rows)
    return rows
