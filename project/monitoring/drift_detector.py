"""
Data & Concept Drift Detector using Population Stability Index (PSI)
and statistical tests (KS-test).  Generates an HTML Evidently report
and logs drift metrics to MLflow.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import mlflow
from loguru import logger
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── PSI ─────────────────────────────────────────────────────────────────────

def _psi_feature(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index for a single continuous feature."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max()) + 1e-9

    breakpoints = np.linspace(min_val, max_val, buckets + 1)

    def _dist(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        freq = counts / len(arr)
        return np.where(freq == 0, 1e-6, freq)

    exp_dist = _dist(expected)
    act_dist = _dist(actual)
    psi = np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))
    return float(psi)


def _psi_label(level: float) -> str:
    if level < 0.1:
        return "stable"
    if level < 0.2:
        return "minor_shift"
    return "major_shift"


# ── KS test ─────────────────────────────────────────────────────────────────

def _ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


# ── Main detector ────────────────────────────────────────────────────────────

class DriftDetector:
    def __init__(self, config: dict):
        self.config = config
        self.psi_threshold = config["monitoring"]["psi_threshold"]
        self.drift_threshold = config["monitoring"]["drift_threshold"]
        os.makedirs("reports", exist_ok=True)

    def _load_reference(self) -> pd.DataFrame:
        ref_path = self.config["monitoring"]["reference_data_path"]
        return pd.read_csv(ref_path)

    def detect(self, current_df: pd.DataFrame, run_id: str | None = None) -> dict:
        reference_df = self._load_reference()

        # Drop target column if present
        feature_cols = [c for c in reference_df.columns if c != "target"]
        ref = reference_df[feature_cols]
        cur = current_df[feature_cols] if "target" in current_df.columns else current_df

        psi_scores: dict[str, float] = {}
        ks_stats:   dict[str, float] = {}
        drifted_features: list[str] = []

        for col in feature_cols:
            psi = _psi_feature(ref[col].values, cur[col].values)
            ks_stat, ks_p = _ks_test(ref[col].values, cur[col].values)

            psi_scores[col] = round(psi, 4)
            ks_stats[col]   = round(ks_stat, 4)

            if psi > self.psi_threshold or ks_p < self.drift_threshold:
                drifted_features.append(col)

        dataset_psi = float(np.mean(list(psi_scores.values())))
        drift_detected = dataset_psi > self.psi_threshold

        report = {
            "dataset_psi":      round(dataset_psi, 4),
            "psi_label":        _psi_label(dataset_psi),
            "drift_detected":   drift_detected,
            "drifted_features": drifted_features,
            "n_drifted":        len(drifted_features),
            "total_features":   len(feature_cols),
            "feature_psi":      psi_scores,
            "feature_ks_stat":  ks_stats,
        }

        report_path = "reports/drift_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Drift report saved → {report_path}")

        if drift_detected:
            logger.warning(
                f"DATA DRIFT DETECTED! Dataset PSI={dataset_psi:.4f} "
                f"(threshold={self.psi_threshold}). "
                f"Drifted features: {drifted_features}"
            )
        else:
            logger.info(f"No significant drift. Dataset PSI={dataset_psi:.4f}")

        if run_id:
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("dataset_psi", dataset_psi)
                mlflow.log_metric("n_drifted_features", len(drifted_features))
                mlflow.log_artifact(report_path)

        # Evidently report (optional — requires evidently installed)
        self._generate_evidently_report(ref, cur)

        return report

    def _generate_evidently_report(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> None:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            ev_report = Report(metrics=[DataDriftPreset()])
            ev_report.run(reference_data=reference, current_data=current)
            ev_report.save_html("reports/evidently_drift_report.html")
            logger.info("Evidently HTML report saved → reports/evidently_drift_report.html")
        except ImportError:
            logger.warning("evidently not installed — skipping HTML report")
        except Exception as e:
            logger.warning(f"Evidently report failed: {e}")


if __name__ == "__main__":
    cfg = load_config()
    detector = DriftDetector(cfg)

    # Simulate current batch = slightly perturbed reference
    ref = pd.read_csv(cfg["monitoring"]["reference_data_path"])
    current = ref.drop(columns=["target"], errors="ignore").copy()
    current += np.random.normal(0, 0.5, current.shape)

    report = detector.detect(current)
    print(json.dumps(report, indent=2))
