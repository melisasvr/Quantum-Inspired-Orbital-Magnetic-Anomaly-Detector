"""
anomaly_ml.py — Magnetic anomaly detection pipeline

Combines Isolation Forest (fast, unsupervised) with a sliding-window
feature extractor to identify statistically significant field residuals
consistent with lithospheric/crustal anomaly signatures.

In production: extend with LSTM autoencoder for temporal patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Feature engineering ────────────────────────────────────────────────────

def _extract_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """
    Compute sliding-window statistical features from magnetic residuals.
    Features chosen to distinguish crustal anomalies from noise/external fields.
    """
    r = df["residual"]

    feats = pd.DataFrame(index=df.index)
    feats["residual"]       = r
    feats["abs_residual"]   = r.abs()
    feats["rolling_mean"]   = r.rolling(window, center=True, min_periods=1).mean()
    feats["rolling_std"]    = r.rolling(window, center=True, min_periods=1).std().fillna(0)
    feats["rolling_max"]    = r.rolling(window, center=True, min_periods=1).max()
    feats["gradient"]       = np.gradient(r.values)
    feats["abs_gradient"]   = np.abs(feats["gradient"])
    feats["curvature"]      = np.gradient(feats["gradient"].values)

    # Spatial context
    feats["lat"]            = df["lat"]
    feats["lon_cos"]        = np.cos(np.radians(df["lon"]))
    feats["lon_sin"]        = np.sin(np.radians(df["lon"]))

    # Deviation from running baseline
    feats["deviation"]      = r - feats["rolling_mean"]

    return feats.fillna(0)


# ── Clustering helpers ─────────────────────────────────────────────────────

def _simple_cluster(mask: np.ndarray, gap: int = 5) -> np.ndarray:
    """
    Label contiguous runs of anomaly=True as distinct clusters,
    merging runs separated by fewer than `gap` samples.
    """
    labels = np.full(len(mask), -1, dtype=int)
    cluster_id = 0
    in_cluster = False
    gap_count  = 0

    for i, flag in enumerate(mask):
        if flag:
            if not in_cluster:
                cluster_id += 1
                in_cluster = True
            labels[i] = cluster_id
            gap_count  = 0
        else:
            if in_cluster:
                gap_count += 1
                if gap_count < gap:
                    labels[i] = cluster_id  # bridge small gaps
                else:
                    in_cluster = False
                    gap_count  = 0

    return labels


# ── Main detector class ────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Two-stage magnetic anomaly detector.

    Stage 1 — Isolation Forest: identifies globally anomalous samples.
    Stage 2 — Threshold filter: keeps only physically meaningful signals
               (|residual| > sigma_threshold × σ_global).
    """

    def __init__(self,
                 contamination: float = 0.05,
                 sigma_threshold: float = 1.5,
                 n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination   = contamination
        self.sigma_threshold = sigma_threshold
        self.n_estimators    = n_estimators
        self.random_state    = random_state

        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators  = n_estimators,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run detection on a Swarm pass DataFrame.

        Returns subset of df rows flagged as anomalies, with extra columns:
            score   : Isolation Forest anomaly score (higher = more anomalous)
            cluster : integer cluster label (1-indexed; -1 = isolated point)
        """
        feats = _extract_features(df)
        X     = self.scaler.fit_transform(feats)

        # Isolation Forest raw score: negative → anomaly
        raw_scores = self.model.fit(X).score_samples(X)
        # Normalise to [0, 1] where 1 = most anomalous
        scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

        # Physical threshold: only keep strong residuals
        sigma = df["residual"].std()
        phys_mask = df["residual"].abs() > self.sigma_threshold * sigma

        # Combined mask
        if_mask  = self.model.predict(X) == -1          # Isolation Forest label
        combined = phys_mask.values & if_mask

        clusters = _simple_cluster(combined)

        result = df.copy()
        result["score"]   = scores
        result["if_flag"] = if_mask
        result["cluster"] = clusters

        anomaly_df = result[result["cluster"] > 0].copy()
        return anomaly_df

    def sensitivity_curve(self, n_points: int = 200) -> pd.DataFrame:
        """
        Return a mock sensitivity vs. signal-amplitude curve for plotting.
        """
        amplitudes = np.logspace(-1, 2.5, n_points)   # 0.1 – 316 nT
        recall     = 1 / (1 + np.exp(-0.15 * (amplitudes - 8)))  # sigmoid
        precision  = 0.55 + 0.42 * (1 - np.exp(-amplitudes / 20))
        return pd.DataFrame({
            "amplitude_nT": amplitudes,
            "recall":       recall,
            "precision":    precision,
        })