"""Regression heads on top of frozen features.

Two heads, because the honest answer to "is the neural head worth it?" is a
measurement, not an assumption:

* :class:`RidgeHead` — closed form, multi-output, alpha chosen on validation. It
  costs milliseconds because the SVD is computed once and reused across the whole
  alpha grid.
* :class:`MLPHead` — a small torch MLP over the same features, early-stopped on
  validation.

Both are multi-task: they predict ``complexity_score`` alongside its two
components. The components are auxiliary supervision — the reported metric is
always the score.
"""
from __future__ import annotations

import time

import numpy as np

from .config import TARGETS, PipelineConfig


class RidgeHead:
    """Multi-output ridge regression solved through one economy SVD.

    For a fixed design matrix the ridge solution at penalty ``a`` is
    ``V diag(s / (s^2 + a)) U^T y``. The factorisation does not depend on ``a``,
    so sweeping the penalty grid is a handful of vector operations rather than a
    refit per value.
    """

    name = "ridge"

    def __init__(self, alphas: tuple[float, ...] = (1.0, 10.0, 100.0)) -> None:
        self.alphas = alphas
        self.alpha: float | None = None
        self.coefficients: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.fit_seconds: float = 0.0

    def _solve(self, u: np.ndarray, s: np.ndarray, vt: np.ndarray,
               y_centered: np.ndarray, alpha: float) -> np.ndarray:
        scale = s / (s**2 + alpha)
        return vt.T @ (scale[:, None] * (u.T @ y_centered))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        primary: int = 0,
    ) -> "RidgeHead":
        started = time.perf_counter()
        self.x_mean = x_train.mean(axis=0)
        self.y_mean = y_train.mean(axis=0)
        x_centered = (x_train - self.x_mean).astype(np.float64)
        y_centered = (y_train - self.y_mean).astype(np.float64)

        u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
        x_validation_centered = (x_validation - self.x_mean).astype(np.float64)

        best = (np.inf, None, None)
        self.alpha_scores: list[tuple[float, float]] = []
        for alpha in self.alphas:
            coefficients = self._solve(u, s, vt, y_centered, alpha)
            prediction = x_validation_centered @ coefficients + self.y_mean
            error = float(np.abs(prediction[:, primary] - y_validation[:, primary]).mean())
            self.alpha_scores.append((alpha, round(error, 4)))
            if error < best[0]:
                best = (error, alpha, coefficients)

        _, self.alpha, self.coefficients = best
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("RidgeHead used before fit()")
        return ((x - self.x_mean).astype(np.float64) @ self.coefficients + self.y_mean).astype(
            np.float32
        )

    def describe(self) -> dict:
        return {
            "head": self.name,
            "alpha": self.alpha,
            "fit_seconds": round(self.fit_seconds, 3),
            "alpha_grid": self.alpha_scores,
        }


class MLPHead:
    """Small multi-task MLP over frozen features, early-stopped on validation."""

    name = "mlp"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.module = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.best_epoch: int = 0
        self.fit_seconds: float = 0.0
        self.history: list[dict] = []

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        primary: int = 0,
    ) -> "MLPHead":
        import torch
        from torch import nn

        started = time.perf_counter()
        torch.manual_seed(self.config.seed)
        config = self.config

        # Targets are standardised so one learning rate suits all three heads.
        self.y_mean = y_train.mean(axis=0)
        self.y_std = np.maximum(y_train.std(axis=0), 1e-6)

        train_x = torch.from_numpy(np.ascontiguousarray(x_train))
        train_y = torch.from_numpy((y_train - self.y_mean) / self.y_std)
        validation_x = torch.from_numpy(np.ascontiguousarray(x_validation))
        validation_y = torch.from_numpy(y_validation)

        weights = torch.tensor(config.target_weights[: y_train.shape[1]], dtype=torch.float32)
        weights = weights / weights.sum()

        self.module = nn.Sequential(
            nn.Linear(train_x.shape[1], config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_hidden, y_train.shape[1]),
        )
        optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=config.mlp_lr, weight_decay=config.mlp_weight_decay
        )
        loss_fn = nn.HuberLoss(reduction="none")

        best_error, best_state, patience = np.inf, None, 0
        # 700 rows x ~2k features is a few megabytes: full-batch steps are faster
        # than minibatching here and make each epoch deterministic.
        for epoch in range(1, config.mlp_epochs + 1):
            self.module.train()
            optimizer.zero_grad()
            loss = (loss_fn(self.module(train_x), train_y).mean(dim=0) * weights).sum()
            loss.backward()
            optimizer.step()

            self.module.eval()
            with torch.no_grad():
                prediction = self.module(validation_x).numpy() * self.y_std + self.y_mean
            error = float(np.abs(prediction[:, primary] - validation_y.numpy()[:, primary]).mean())
            self.history.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_mae": error})

            if error < best_error - 1e-5:
                best_error, self.best_epoch, patience = error, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
            else:
                patience += 1
                if patience >= config.mlp_patience:
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.module is None:
            raise RuntimeError("MLPHead used before fit()")
        self.module.eval()
        with torch.no_grad():
            output = self.module(torch.from_numpy(np.ascontiguousarray(x))).numpy()
        return (output * self.y_std + self.y_mean).astype(np.float32)

    def describe(self) -> dict:
        return {
            "head": self.name,
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.history),
            "fit_seconds": round(self.fit_seconds, 3),
            "hidden": self.config.mlp_hidden,
        }


class MeanBaseline:
    """Predict the training mean. The floor any real head has to clear."""

    name = "mean"

    def __init__(self) -> None:
        self.y_mean: np.ndarray | None = None
        self.fit_seconds = 0.0

    def fit(self, x_train, y_train, x_validation, y_validation, *, primary: int = 0):
        self.y_mean = y_train.mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(self.y_mean[None, :], len(x), axis=0).astype(np.float32)

    def describe(self) -> dict:
        return {"head": self.name, "fit_seconds": 0.0}
