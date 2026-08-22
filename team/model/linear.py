"""Minimal linear models (numpy-free): normal equations with ridge."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*m)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, mid, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(mid):
            for j in range(cols):
                out[i][j] += a[i][k] * b[k][j]
    return out


def _solve_linear(xtx: list[list[float]], xty: list[list[float]]) -> list[list[float]]:
    """Solve xtx W = xty via Gauss-Jordan (small matrices only)."""
    n = len(xtx)
    aug = [xtx[i][:] + xty[i][:] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col] or 1e-12
        aug[col] = [v / div for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    return [[aug[i][n]] for i in range(n)]


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: list[list[float]] | None = None  # (features+1) x outputs

    def fit(self, x: list[list[float]], y: list[list[float]]) -> "RidgeRegressor":
        n_feat = len(x[0])
        n_out = len(y[0])
        x_aug = [row + [1.0] for row in x]
        xt = _transpose(x_aug)
        xtx = _matmul(xt, x_aug)
        for i in range(n_feat + 1):
            xtx[i][i] += self.alpha
        self.weights = [[0.0] * n_out for _ in range(n_feat + 1)]
        for j in range(n_out):
            col_y = [[row[j]] for row in y]
            xty = _matmul(xt, col_y)
            col_w = _solve_linear(xtx, xty)
            for i in range(n_feat + 1):
                self.weights[i][j] = col_w[i][0]
        return self

    def predict(self, x: list[list[float]]) -> list[list[float]]:
        assert self.weights is not None
        n_out = len(self.weights[0])
        out = []
        for row in x:
            aug = row + [1.0]
            pred = [sum(aug[i] * self.weights[i][j] for i in range(len(aug))) for j in range(n_out)]
            out.append(pred)
        return out

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "weights": self.weights}

    @classmethod
    def from_dict(cls, d: dict) -> "RidgeRegressor":
        m = cls(alpha=d["alpha"])
        m.weights = d["weights"]
        return m


class SoftmaxRouter:
    """Multinomial logistic regression via batch gradient descent."""

    def __init__(self, classes: list[str], lr: float = 0.05, epochs: int = 800, l2: float = 1e-3):
        self.classes = classes
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[list[float]] | None = None  # n_feat x n_class

    def _softmax(self, logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def fit(self, x: list[list[float]], y_idx: list[int]) -> "SoftmaxRouter":
        n_feat = len(x[0])
        n_class = len(self.classes)
        w = [[0.0] * n_class for _ in range(n_feat)]
        for _ in range(self.epochs):
            for row, target in zip(x, y_idx):
                logits = [sum(row[i] * w[i][c] for i in range(n_feat)) for c in range(n_class)]
                probs = self._softmax(logits)
                for i in range(n_feat):
                    for c in range(n_class):
                        grad = row[i] * (probs[c] - (1.0 if c == target else 0.0))
                        w[i][c] -= self.lr * (grad + self.l2 * w[i][c])
        self.weights = w
        return self

    def predict(self, x: list[list[float]]) -> list[str]:
        assert self.weights is not None
        out = []
        for row in x:
            logits = [
                sum(row[i] * self.weights[i][c] for i in range(len(row)))
                for c in range(len(self.classes))
            ]
            out.append(self.classes[max(range(len(logits)), key=lambda i: logits[i])])
        return out

    def predict_proba(self, x: list[list[float]]) -> list[list[float]]:
        assert self.weights is not None
        out = []
        for row in x:
            logits = [
                sum(row[i] * self.weights[i][c] for i in range(len(row)))
                for c in range(len(self.classes))
            ]
            out.append(self._softmax(logits))
        return out

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "lr": self.lr,
            "epochs": self.epochs,
            "l2": self.l2,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoftmaxRouter":
        m = cls(classes=d["classes"], lr=d["lr"], epochs=d["epochs"], l2=d["l2"])
        m.weights = d["weights"]
        return m


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
