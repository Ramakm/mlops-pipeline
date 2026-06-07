"""
Model Evaluation — generates a full test-set report: classification
metrics, confusion matrix, ROC curve, and feature importance.
Saves artefacts to reports/ and logs them to MLflow.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(config: dict, run_id: str | None = None) -> dict:
    processed_path = config["data"]["processed_path"]
    os.makedirs("reports", exist_ok=True)

    X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).squeeze()

    model = joblib.load("models/logistic_regression.joblib")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy":  accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall":    recall_score(y_test, y_pred, zero_division=0),
        "test_f1":        f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc":   roc_auc_score(y_test, y_prob),
    }

    logger.info("=== Test Set Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save metrics JSON
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=["malignant", "benign"]
    )
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["malignant", "benign"],
        yticklabels=["malignant", "benign"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc']:.3f}", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=150)
    plt.close()

    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        "feature": X_test.columns,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv("reports/feature_importance.csv", index=False)

    # Log everything back to MLflow if run_id given
    if run_id:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            for art in [
                "reports/metrics.json",
                "reports/classification_report.txt",
                "reports/confusion_matrix.png",
                "reports/roc_curve.png",
                "reports/feature_importance.csv",
            ]:
                mlflow.log_artifact(art)
        logger.info(f"Evaluation artefacts logged to MLflow run {run_id}")

    return metrics


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
