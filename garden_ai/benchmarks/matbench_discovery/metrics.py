"""Functions to classify energy above convex hull predictions as true/false
positive/negative and compute performance metrics.

Adapted from matbench-discovery to avoid import issues.
Original source: https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/metrics/discovery.py
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Default stability threshold from matbench-discovery
# STABILITY_THRESHOLD = 0.0


def classify_stable(
    each_true: Sequence[float] | pd.Series | np.ndarray,
    each_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = 0.0,
    fillna: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Classify model stability predictions as true/false positive/negatives (usually
    w.r.t DFT-ground truth labels). All energies are assumed to be in eV/atom
    (but shouldn't really matter as long as they're consistent).

    Args:
        each_true (Sequence[float] | pd.Series): Ground truth energy above convex hull
            values.
        each_pred (Sequence[float] | pd.Series): Model-predicted energy above convex
            hull values.
        stability_threshold (float, optional): Maximum energy above convex hull
            for a material to still be considered stable. Usually 0, 0.05 or 0.1.
            Defaults to 0.0, meaning a material has to be directly on
            the hull to be called stable. Negative values mean a material has to pull
            the known hull down by that amount to count as stable. Few materials lie
            below the known hull, so only negative values very close to 0 make sense.
        fillna (bool): Whether to fill NaNs as the model predicting unstable. Defaults
            to True.

    Returns:
        tuple[TP, FN, FP, TN]: Indices as pd.Series for true positives,
            false negatives, false positives and true negatives (in this order).

    Raises:
        ValueError: If sum of positive + negative preds doesn't add up to the total.
    """
    if len(each_true) != len(each_pred):
        raise ValueError(f"{len(each_true)=} != {len(each_pred)=}")

    each_true_arr, each_pred_arr = pd.Series(each_true), pd.Series(each_pred)

    if stability_threshold is None or np.isnan(stability_threshold):
        raise ValueError("stability_threshold must be a real number")
    actual_pos = each_true_arr <= (stability_threshold or 0)
    actual_neg = each_true_arr > (stability_threshold or 0)

    model_pos = each_pred_arr <= (stability_threshold or 0)
    model_neg = each_pred_arr > (stability_threshold or 0)

    if fillna:
        nan_mask = np.isnan(each_pred)
        # for in both the model's stable and unstable preds, fill NaNs as unstable
        model_pos[nan_mask] = False
        model_neg[nan_mask] = True

        n_pos, n_neg, total = model_pos.sum(), model_neg.sum(), len(each_pred)
        if n_pos + n_neg != total:
            raise ValueError(
                f"after filling NaNs, the sum of positive ({n_pos}) and negative "
                f"({n_neg}) predictions should add up to {total=}"
            )

    true_pos = actual_pos & model_pos
    false_neg = actual_pos & model_neg
    false_pos = actual_neg & model_pos
    true_neg = actual_neg & model_neg

    return true_pos, false_neg, false_pos, true_neg


def stable_metrics(
    each_true: Sequence[float] | pd.Series | np.ndarray,
    each_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    stability_threshold: float = 0.0,
    fillna: bool = True,
    prevalence: float | None = None,
) -> dict[str, float]:
    """Get a dictionary of stability prediction metrics. Mostly binary classification
    metrics, but also MAE, RMSE and R2.

    Args:
        each_true (Sequence[float] | pd.Series): true energy above convex hull
        each_pred (Sequence[float] | pd.Series): predicted energy above convex hull
        stability_threshold (float): Where to place stability threshold relative to
            convex hull in eV/atom, usually 0 or 0.1 eV. Default = 0.0.
        fillna (bool): Whether to fill NaNs as the model predicting unstable. Defaults
            to True.
        prevalence (float, optional): Prevalence of stable materials in the dataset.
            If None, calculated from the input data. Defaults to None.

    Note: Should give equivalent classification metrics to
        sklearn.metrics.classification_report(
            each_true > stability_threshold,
            each_pred > stability_threshold,
            output_dict=True,
        )
        when using the same stability_threshold.

    Returns:
        dict[str, float]: dictionary of classification metrics with keys DAF, Precision,
            Recall, Accuracy, F1, TPR, FPR, TNR, FNR, MAE, RMSE, R2.

    Raises:
        ValueError: If FPR + TNR don't add up to 1.
        ValueError: If TPR + FNR don't add up to 1.
    """
    n_true_pos, n_false_neg, n_false_pos, n_true_neg = map(
        sum,
        classify_stable(
            each_true, each_pred, stability_threshold=stability_threshold, fillna=fillna
        ),
    )

    n_total_pos = n_true_pos + n_false_neg
    n_total_neg = n_true_neg + n_false_pos
    # prevalence: dummy discovery rate of stable crystals by selecting randomly from
    # all materials
    if prevalence is None:
        prevalence = (
            n_total_pos / (n_total_pos + n_total_neg)
            if (n_total_pos + n_total_neg) > 0
            else float("nan")
        )
    # Calculate ratios with guards against division by zero
    precision = (
        n_true_pos / (n_true_pos + n_false_pos)
        if (n_true_pos + n_false_pos) > 0
        else float("nan")
    )
    recall = n_true_pos / n_total_pos if n_total_pos > 0 else float("nan")

    TPR = recall
    FPR = n_false_pos / n_total_neg if n_total_neg > 0 else float("nan")
    TNR = n_true_neg / n_total_neg if n_total_neg > 0 else float("nan")
    FNR = n_false_neg / n_total_pos if n_total_pos > 0 else float("nan")

    # sanity check: false positives + true negatives = all negatives
    if FPR > 0 and TNR > 0 and FPR + TNR != 1:
        # Floating point tolerance
        if abs(FPR + TNR - 1) > 1e-6:
            raise ValueError(f"{FPR=} {TNR=} don't add up to 1")

    # sanity check: true positives + false negatives = all positives
    if TPR > 0 and FNR > 0 and TPR + FNR != 1:
        # Floating point tolerance
        if abs(TPR + FNR - 1) > 1e-6:
            raise ValueError(f"{TPR=} {FNR=} don't add up to 1")

    # Drop NaNs to calculate regression metrics
    is_nan = np.isnan(each_true) | np.isnan(each_pred)
    each_true, each_pred = np.array(each_true)[~is_nan], np.array(each_pred)[~is_nan]

    if precision + recall == 0:  # Calculate F1 score, handling division by zero
        f1_score = float("nan")
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return dict(
        F1=f1_score,
        DAF=precision / prevalence if prevalence > 0 else float("nan"),
        Precision=precision,
        Recall=recall,
        Accuracy=(
            (n_true_pos + n_true_neg) / (n_total_pos + n_total_neg)
            if (n_total_pos + n_total_neg > 0)
            else float("nan")
        ),
        TPR=TPR,
        FPR=FPR,
        TNR=TNR,
        FNR=FNR,
        TP=n_true_pos,
        FP=n_false_pos,
        TN=n_true_neg,
        FN=n_false_neg,
        MAE=np.abs(each_true - each_pred).mean(),
        RMSE=((each_true - each_pred) ** 2).mean() ** 0.5,
        R2=r2_score(each_true, each_pred) if len(each_true) > 1 else float("nan"),
    )
