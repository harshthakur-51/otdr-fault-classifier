"""A small fully connected classifier implemented without external dependencies."""

from __future__ import annotations

import math
import random


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


class NeuralClassifier:
    def __init__(self, inputs: int, hidden: int, outputs: int, seed: int = 12) -> None:
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.35, 0.35) for _ in range(inputs)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-0.35, 0.35) for _ in range(hidden)] for _ in range(outputs)]
        self.b2 = [0.0] * outputs

    def _forward(self, row: list[float]) -> tuple[list[float], list[float]]:
        hidden = [math.tanh(sum(weight * value for weight, value in zip(weights, row)) + bias) for weights, bias in zip(self.w1, self.b1)]
        scores = [sum(weight * value for weight, value in zip(weights, hidden)) + bias for weights, bias in zip(self.w2, self.b2)]
        return hidden, _softmax(scores)

    def train(self, rows: list[list[float]], targets: list[int], epochs: int = 80, rate: float = 0.035, seed: int = 22) -> None:
        rng = random.Random(seed)
        order = list(range(len(rows)))
        for _ in range(epochs):
            rng.shuffle(order)
            for index in order:
                row = rows[index]
                hidden, probabilities = self._forward(row)
                output_delta = probabilities[:]
                output_delta[targets[index]] -= 1.0
                hidden_delta = [(1.0 - value * value) * sum(output_delta[k] * self.w2[k][j] for k in range(len(self.w2))) for j, value in enumerate(hidden)]
                for k in range(len(self.w2)):
                    for j in range(len(hidden)):
                        self.w2[k][j] -= rate * output_delta[k] * hidden[j]
                    self.b2[k] -= rate * output_delta[k]
                for j in range(len(self.w1)):
                    for i in range(len(row)):
                        self.w1[j][i] -= rate * hidden_delta[j] * row[i]
                    self.b1[j] -= rate * hidden_delta[j]

    def predict(self, row: list[float]) -> int:
        return max(enumerate(self._forward(row)[1]), key=lambda item: item[1])[0]


def accuracy(model: NeuralClassifier, rows: list[list[float]], targets: list[int]) -> float:
    correct = sum(model.predict(row) == target for row, target in zip(rows, targets))
    return correct / len(rows)


def confusion_matrix(model: NeuralClassifier, rows: list[list[float]], targets: list[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for row, target in zip(rows, targets):
        matrix[target][model.predict(row)] += 1
    return matrix
