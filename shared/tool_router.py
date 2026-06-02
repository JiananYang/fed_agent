from __future__ import annotations

import math
from dataclasses import dataclass


LABELS = ["calculator", "paper_search", "claim_verification", "general"]

VOCAB = [
    "sum",
    "calculate",
    "mean",
    "total",
    "sample",
    "participants",
    "measurement",
    "search",
    "find",
    "look",
    "paper",
    "papers",
    "study",
    "studies",
    "evidence",
    "abstract",
    "claim",
    "verify",
    "check",
    "support",
    "supports",
    "refute",
    "refutes",
    "biomedical",
    "scientific",
    "hello",
    "explain",
    "help",
]


def featurize(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if token in lowered else 0.0 for token in VOCAB] + [1.0]


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class RouterPrediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]


class ToolRouter:
    """Tiny trainable tool selector for dependency-free prototyping."""

    def __init__(self) -> None:
        rows = len(VOCAB) + 1
        cols = len(LABELS)
        self.weights = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def get_parameters(self) -> list[list[float]]:
        return [row[:] for row in self.weights]

    def set_parameters(self, parameters: list[list[float]]) -> None:
        self.weights = [row[:] for row in parameters]

    def predict(self, text: str) -> RouterPrediction:
        x = featurize(text)
        logits = []
        for label_idx in range(len(LABELS)):
            logits.append(sum(x[i] * self.weights[i][label_idx] for i in range(len(x))))

        probs = softmax(logits)
        ranked = sorted(zip(LABELS, probs), key=lambda item: item[1], reverse=True)
        return RouterPrediction(label=ranked[0][0], confidence=ranked[0][1], top_k=ranked[:3])

    def train(self, examples: list[tuple[str, str]], epochs: int = 30, lr: float = 0.25) -> float:
        if not examples:
            return 0.0

        last_loss = 0.0
        for _ in range(epochs):
            total_loss = 0.0
            for query, label in examples:
                x = featurize(query)
                y = LABELS.index(label)
                logits = []
                for label_idx in range(len(LABELS)):
                    logits.append(sum(x[i] * self.weights[i][label_idx] for i in range(len(x))))

                probs = softmax(logits)
                total_loss += -math.log(max(probs[y], 1e-9))

                for i, value in enumerate(x):
                    for label_idx in range(len(LABELS)):
                        target = 1.0 if label_idx == y else 0.0
                        gradient = value * (probs[label_idx] - target)
                        self.weights[i][label_idx] -= lr * gradient

            last_loss = total_loss / len(examples)

        return last_loss


def average_parameters(parameter_sets: list[list[list[float]]]) -> list[list[float]]:
    rows = len(parameter_sets[0])
    cols = len(parameter_sets[0][0])
    averaged = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for params in parameter_sets:
        for row in range(rows):
            for col in range(cols):
                averaged[row][col] += params[row][col] / len(parameter_sets)
    return averaged
