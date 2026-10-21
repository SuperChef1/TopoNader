from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr


@dataclass
class TopoNaderEvaluation:
    sample_count: int
    rmse: float
    mae: float
    r2: float
    kendall_tau: float
    spearman_rho: float
    pairwise_accuracy: float
    topk_recall: float
    topk_regret: float
    uncertainty_error_correlation: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_correlation(value: float | np.ndarray) -> float:
    scalar = float(np.asarray(value).reshape(-1)[0])
    return scalar if np.isfinite(scalar) else 0.0


def pairwise_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    correct = 0
    comparable = 0
    for left in range(truth.size):
        for right in range(left + 1, truth.size):
            true_delta = truth[left] - truth[right]
            pred_delta = prediction[left] - prediction[right]
            if true_delta == 0:
                continue
            comparable += 1
            correct += int(true_delta * pred_delta > 0)
    return float(correct / comparable) if comparable else 1.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None = None,
    topk: int = 3,
) -> TopoNaderEvaluation:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or truth.size < 2:
        raise ValueError("evaluation requires matching arrays with at least two samples")
    residual = truth - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
    tau = _safe_correlation(kendalltau(truth, prediction).statistic)
    rho = _safe_correlation(spearmanr(truth, prediction).statistic)
    k = max(1, min(int(topk), truth.size))
    true_top = set(np.argsort(-truth)[:k].tolist())
    predicted_top = set(np.argsort(-prediction)[:k].tolist())
    recall = len(true_top & predicted_top) / k
    best_true = float(np.max(truth))
    best_selected = float(np.max(truth[list(predicted_top)]))
    regret = (best_true - best_selected) / max(abs(best_true), 1e-12)
    uncertainty_error = None
    if uncertainty is not None:
        sigma = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if sigma.shape != truth.shape:
            raise ValueError("uncertainty shape does not match predictions")
        if np.std(sigma) < 1e-12 or np.std(np.abs(residual)) < 1e-12:
            uncertainty_error = 0.0
        else:
            uncertainty_error = _safe_correlation(spearmanr(sigma, np.abs(residual)).statistic)
    return TopoNaderEvaluation(
        sample_count=int(truth.size),
        rmse=rmse,
        mae=mae,
        r2=r2,
        kendall_tau=tau,
        spearman_rho=rho,
        pairwise_accuracy=pairwise_ranking_accuracy(truth, prediction),
        topk_recall=float(recall),
        topk_regret=float(max(0.0, regret)),
        uncertainty_error_correlation=uncertainty_error,
    )
