"""Command-line training run for the OTDR classifier."""

from __future__ import annotations

import argparse

from .features import extract_features, normalize
from .model import NeuralClassifier, accuracy, confusion_matrix
from .traces import FAULT_CLASSES, generate_dataset


def run(samples_per_class: int, epochs: int, seed: int) -> float:
    dataset = generate_dataset(samples_per_class, seed)
    split = int(len(dataset) * 0.8)
    training, testing = dataset[:split], dataset[split:]
    train_x_raw = [extract_features(trace) for trace, _ in training]
    test_x_raw = [extract_features(trace) for trace, _ in testing]
    train_x = normalize(train_x_raw, train_x_raw)
    test_x = normalize(train_x_raw, test_x_raw)
    train_y = [FAULT_CLASSES.index(label) for _, label in training]
    test_y = [FAULT_CLASSES.index(label) for _, label in testing]
    model = NeuralClassifier(len(train_x[0]), 18, len(FAULT_CLASSES))
    model.train(train_x, train_y, epochs=epochs)
    score = accuracy(model, test_x, test_y)
    print("OTDR fault classifier evaluation")
    print(f"Samples: {len(dataset)} | Test accuracy: {score:.2%}")
    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print(" " * 19 + " ".join(f"{i:>3}" for i in range(len(FAULT_CLASSES))))
    for label, row in zip(FAULT_CLASSES, confusion_matrix(model, test_x, test_y, len(FAULT_CLASSES))):
        print(f"{label[:18]:<18} " + " ".join(f"{value:>3}" for value in row))
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.samples_per_class, args.epochs, args.seed)


if __name__ == "__main__":
    main()
