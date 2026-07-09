"""Evaluation metrics for fraud detection.

Accuracy is useless here: predicting "never fraud" scores 98%+ on a 1.5% base
rate. Even ROC-AUC flatters a model on imbalanced data, because the vast
negative class makes the false-positive rate look small however many genuine
customers get blocked.

So the headline metrics are PR-AUC and, above all, **expected business cost**:

    cost = sum(amount of each missed fraud) + (false positives x cost of a block)

The asymmetry is the point. A missed fraud costs the chargeback -- the actual
transaction amount, which varies per transaction. A false positive costs a
declined payment and some goodwill, which is roughly fixed. Optimising F1
implicitly assumes those two errors cost the same. They do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: What it costs the business to wrongly block one genuine transaction: the lost
#: margin on the sale, the support contact, and a slice of churn risk. A single
#: number is a simplification, but a defensible one -- and it is a dial the
#: fraud team can turn, which is the point of naming it explicitly.
DEFAULT_FALSE_POSITIVE_COST = 5.0

#: Used only when per-transaction amounts are unavailable.
DEFAULT_FALSE_NEGATIVE_COST = 100.0


@dataclass(frozen=True)
class CostModel:
    """Prices the two ways a fraud model can be wrong.

    Attributes:
        false_positive_cost: Cost of blocking one genuine transaction.
        false_negative_cost: Fallback cost of one missed fraud, used only when
            transaction amounts are not supplied. When they are, a missed fraud
            costs its own amount.
    """

    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST

    def __post_init__(self) -> None:
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("costs must be non-negative")


@dataclass(frozen=True)
class Evaluation:
    """Everything we compare two model variants on."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    total_cost: float
    cost_per_1000: float
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, float]:
        """Flat mapping, ready to log to Vertex AI Experiments."""
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "total_cost": self.total_cost,
            "cost_per_1000": self.cost_per_1000,
            "false_positives": float(self.false_positives),
            "false_negatives": float(self.false_negatives),
            "true_positives": float(self.true_positives),
        }


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must contain only 0 and 1")
    return y_true.astype(int), y_score


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    amounts: np.ndarray | None = None,
    cost_model: CostModel | None = None,
) -> float:
    """Total business cost of a set of hard 0/1 predictions.

    A missed fraud (false negative) costs the transaction amount when `amounts`
    is supplied, else `cost_model.false_negative_cost`. A blocked genuine
    transaction (false positive) costs `cost_model.false_positive_cost`.
    Correct predictions cost nothing.
    """
    cost_model = cost_model or CostModel()
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    false_negative = (y_true == 1) & (y_pred == 0)
    false_positive = (y_true == 0) & (y_pred == 1)

    if amounts is None:
        fn_cost = float(false_negative.sum()) * cost_model.false_negative_cost
    else:
        amounts = np.asarray(amounts, dtype=float)
        if amounts.shape != y_true.shape:
            raise ValueError("amounts must have the same shape as y_true")
        fn_cost = float(amounts[false_negative].sum())

    fp_cost = float(false_positive.sum()) * cost_model.false_positive_cost
    return fn_cost + fp_cost


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    amounts: np.ndarray | None = None,
    cost_model: CostModel | None = None,
) -> float:
    """The decision threshold that minimises business cost on this data.

    Searches the observed scores rather than a fixed grid, so the returned
    threshold is always achievable. Ties break toward the higher threshold,
    which blocks fewer customers for the same cost.

    Note this threshold is fitted *on the data it is evaluated against*. Picking
    it on the test set would be optimistic; callers fit it on validation data.
    """
    y_true, y_score = _validate(y_true, y_score)

    # Candidate thresholds: just above each distinct score, plus one that
    # predicts everything negative.
    candidates = np.unique(y_score)
    candidates = np.concatenate([candidates, [np.nextafter(candidates.max(), np.inf)]])

    best_threshold = float(candidates[-1])
    best_cost = np.inf
    for threshold in candidates:
        cost = business_cost(
            y_true, (y_score >= threshold).astype(int), amounts=amounts, cost_model=cost_model
        )
        if cost < best_cost or (cost == best_cost and threshold > best_threshold):
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    amounts: np.ndarray | None = None,
    cost_model: CostModel | None = None,
) -> Evaluation:
    """Score predicted fraud probabilities at a given decision threshold."""
    y_true, y_score = _validate(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    n_classes = len(np.unique(y_true))
    # roc_auc_score and average_precision_score are undefined with one class.
    roc = float(roc_auc_score(y_true, y_score)) if n_classes == 2 else float("nan")
    pr = float(average_precision_score(y_true, y_score)) if n_classes == 2 else float("nan")

    total = business_cost(y_true, y_pred, amounts=amounts, cost_model=cost_model)

    return Evaluation(
        roc_auc=roc,
        pr_auc=pr,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
        total_cost=total,
        cost_per_1000=total / len(y_true) * 1000.0,
        false_positives=int(((y_true == 0) & (y_pred == 1)).sum()),
        false_negatives=int(((y_true == 1) & (y_pred == 0)).sum()),
        true_positives=int(((y_true == 1) & (y_pred == 1)).sum()),
    )


def bootstrap_cost_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    threshold_a: float,
    threshold_b: float,
    amounts: np.ndarray | None = None,
    cost_model: CostModel | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for `cost_per_1000(A) - cost_per_1000(B)`.

    A single point estimate cannot tell you whether XGBoost genuinely beats
    LightGBM or whether it won by luck on a handful of large frauds. Resampling
    the *same* transactions for both variants keeps the comparison paired.

    Returns:
        `(point_estimate, lower_bound, upper_bound)`. If the interval straddles
        zero, the two variants are not distinguishable on this test set.
    """
    y_true, score_a = _validate(y_true, score_a)
    _, score_b = _validate(y_true, score_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def cost_per_1000(idx: np.ndarray, score: np.ndarray, threshold: float) -> float:
        subset_amounts = None if amounts is None else np.asarray(amounts)[idx]
        cost = business_cost(
            y_true[idx],
            (score[idx] >= threshold).astype(int),
            amounts=subset_amounts,
            cost_model=cost_model,
        )
        return cost / len(idx) * 1000.0

    full = np.arange(n)
    point = cost_per_1000(full, score_a, threshold_a) - cost_per_1000(full, score_b, threshold_b)

    differences = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        differences[i] = cost_per_1000(idx, score_a, threshold_a) - cost_per_1000(
            idx, score_b, threshold_b
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(differences, alpha))
    upper = float(np.quantile(differences, 1.0 - alpha))
    return float(point), lower, upper
