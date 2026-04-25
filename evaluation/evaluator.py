from __future__ import annotations

import os

import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc

from evaluation.summary import EvalReport
from training.config import MLflowRunRef


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        log_loss,
        accuracy_score,
    )
    pred_arr = np.array(y_pred)

    # Determine whether predictions look like probabilities or class labels.
    # A pyfunc model may return a DataFrame; handle that.
    if isinstance(pred_arr[0], (list, np.ndarray)) or (
        hasattr(y_pred, "ndim") and y_pred.ndim == 2 and y_pred.shape[1] > 1
    ):
        proba = pred_arr[:, 1] if pred_arr.shape[1] == 2 else pred_arr
        labels = (proba >= 0.5).astype(int)
    elif np.all((pred_arr >= 0) & (pred_arr <= 1)):
        # Could be probabilities for the positive class
        proba = pred_arr
        labels = (proba >= 0.5).astype(int)
    else:
        labels = pred_arr.astype(int)
        proba = labels  # AUC won't be meaningful but won't crash

    y_true_arr = np.array(y_true)

    try:
        auc = roc_auc_score(y_true_arr, proba)
    except Exception:
        auc = float("nan")

    try:
        ll = log_loss(y_true_arr, proba)
    except Exception:
        ll = float("nan")

    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true_arr, labels, average="macro", zero_division=0)),
        "precision": float(
            precision_score(y_true_arr, labels, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, labels, average="macro", zero_division=0)
        ),
        "log_loss": float(ll),
        "accuracy": float(accuracy_score(y_true_arr, labels)),
    }


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    nonzero = y_true_arr != 0
    mape = (
        float(np.mean(np.abs((y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero])))
        if nonzero.any()
        else float("nan")
    )

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    if task_type == "classification":
        return _classification_metrics(y_true, y_pred)
    return _regression_metrics(y_true, y_pred)


def evaluate_model(
    mlflow_run_ref: MLflowRunRef,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    task_type: str,
    run_id: str,
) -> EvalReport:
    min_improvement = float(os.getenv("EVAL_MIN_IMPROVEMENT", "0.005"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "watchtower-champion")

    new_model = mlflow.pyfunc.load_model(mlflow_run_ref.artifact_uri)
    new_preds = np.array(new_model.predict(X_eval))
    new_metrics = _compute_metrics(y_eval, new_preds, task_type)

    champion_metrics: dict[str, float] = {}
    promote_recommended: bool = False

    try:
        champion = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        champ_preds = np.array(champion.predict(X_eval))
        champion_metrics = _compute_metrics(y_eval, champ_preds, task_type)
    except mlflow.exceptions.MlflowException:
        # No champion registered yet — first run
        promote_recommended = True

    delta_metrics: dict[str, float] = {
        k: new_metrics[k] - champion_metrics.get(k, 0.0) for k in new_metrics
    }

    if not promote_recommended:
        if task_type == "classification":
            promote_recommended = delta_metrics.get("auc", 0.0) > min_improvement
        else:
            # Lower RMSE is better; improvement means delta_rmse is negative and large enough
            promote_recommended = delta_metrics.get("rmse", 0.0) < -min_improvement

    primary_metric = "auc" if task_type == "classification" else "rmse"

    return EvalReport(
        run_id=run_id,
        new_model_metrics=new_metrics,
        champion_metrics=champion_metrics,
        delta_metrics=delta_metrics,
        promote_recommended=promote_recommended,
        primary_metric=primary_metric,
        n_eval_samples=len(X_eval),
    )
