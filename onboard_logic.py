"""
onboard_logic.py — Autonomous onboard anomaly triage

Simulates the edge decision-making layer that would run on a flight
computer (e.g., ARM Cortex-M7) to triage detected anomalies and
assign downlink priority without ground intervention.

Priority tiers:
  CRITICAL (3) — Strong anomaly + quantum-confirmed + novel location
  HIGH     (2) — Strong anomaly or known anomaly with new feature
  MEDIUM   (1) — Candidate anomaly, quantum-marginal
  NOMINAL  (0) — No action required
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ── Known anomaly catalog (simplified) ────────────────────────────────────

KNOWN_ANOMALIES = pd.DataFrame([
    {"name": "Bangui",         "lat":  5.0, "lon":  23.0, "radius_deg": 5.0},
    {"name": "Kursk",          "lat": 52.0, "lon":  37.5, "radius_deg": 4.0},
    {"name": "South Africa",   "lat":-30.0, "lon":  25.0, "radius_deg": 6.0},
    {"name": "Japan Arc",      "lat": 35.0, "lon": 135.0, "radius_deg": 4.0},
    {"name": "Antarctic Pen.", "lat":-55.0, "lon": -65.0, "radius_deg": 5.0},
    {"name": "East Australia", "lat":-28.0, "lon": 147.0, "radius_deg": 5.0},
])


def _nearest_known(lat: float, lon: float) -> tuple:
    """Return (name, distance_deg) of nearest known anomaly."""
    dists = np.sqrt(
        (KNOWN_ANOMALIES["lat"] - lat)**2 +
        (KNOWN_ANOMALIES["lon"] - lon)**2
    )
    idx = dists.idxmin()
    return KNOWN_ANOMALIES.loc[idx, "name"], float(dists[idx])


def _in_known_catalog(lat: float, lon: float) -> bool:
    """True if (lat, lon) falls within a known anomaly footprint."""
    for _, row in KNOWN_ANOMALIES.iterrows():
        dist = np.sqrt((row["lat"] - lat)**2 + (row["lon"] - lon)**2)
        if dist <= row["radius_deg"]:
            return True
    return False


# ── Decision rules ────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    cluster_id:        int
    center_lat:        float
    center_lon:        float
    peak_residual_nT:  float
    anomaly_score:     float
    known_anomaly:     bool
    nearest_catalog:   str
    dist_to_catalog:   float
    quantum_confirmed: bool
    priority_level:    int          # 0–3
    priority_label:    str
    downlink_priority: bool
    rationale:         str


PRIORITY_LABELS = {3: "CRITICAL", 2: "HIGH", 1: "MEDIUM", 0: "NOMINAL"}


class OnboardDecisionLogic:
    """
    Rule-based triage engine for detected magnetic anomaly clusters.

    Thresholds are tunable for mission objectives (e.g., mineral survey
    vs. space weather monitoring vs. earthquake precursor study).
    """

    def __init__(self,
                 score_critical:  float = 0.85,
                 score_high:      float = 0.70,
                 score_medium:    float = 0.50,
                 snr_gain_db_min: float = 10.0,
                 novelty_dist_deg: float = 3.0):
        self.score_critical   = score_critical
        self.score_high       = score_high
        self.score_medium     = score_medium
        self.snr_gain_db_min  = snr_gain_db_min
        self.novelty_dist_deg = novelty_dist_deg

    def _classify_cluster(self, cluster_df: pd.DataFrame,
                           qresults: dict) -> DecisionRecord:
        cid    = int(cluster_df["cluster"].iloc[0])
        clat   = float(cluster_df["lat"].mean())
        clon   = float(cluster_df["lon"].mean())
        score  = float(cluster_df["score"].max())
        peak_r = float(cluster_df["residual"].abs().max())

        known = _in_known_catalog(clat, clon)
        nearest_name, nearest_dist = _nearest_known(clat, clon)

        q_confirmed = qresults["snr_gain_db"] >= self.snr_gain_db_min
        is_novel    = nearest_dist > self.novelty_dist_deg

        # ── Priority assignment ──────────────────────────────────
        if score >= self.score_critical and q_confirmed and is_novel:
            level    = 3
            rationale = (f"Novel location ({nearest_dist:.1f}° from {nearest_name}), "
                         f"quantum-confirmed SNR +{qresults['snr_gain_db']:.1f} dB, "
                         f"score={score:.2f}")
        elif score >= self.score_high and q_confirmed:
            level    = 2
            rationale = (f"High-confidence anomaly (score={score:.2f}), "
                         f"peak={peak_r:.1f} nT, near {nearest_name}")
        elif score >= self.score_medium:
            level    = 1
            rationale = (f"Candidate anomaly (score={score:.2f}), "
                         f"marginal quantum confirmation")
        else:
            level    = 0
            rationale = "Below detection thresholds — nominal field variation"

        return DecisionRecord(
            cluster_id       = cid,
            center_lat       = clat,
            center_lon       = clon,
            peak_residual_nT = peak_r,
            anomaly_score    = score,
            known_anomaly    = known,
            nearest_catalog  = nearest_name,
            dist_to_catalog  = nearest_dist,
            quantum_confirmed= q_confirmed,
            priority_level   = level,
            priority_label   = PRIORITY_LABELS[level],
            downlink_priority= level >= 2,
            rationale        = rationale,
        )

    def evaluate(self, anomaly_df: pd.DataFrame,
                 qresults: dict) -> pd.DataFrame:
        """
        Evaluate all detected anomaly clusters.

        Returns a DataFrame of DecisionRecord instances, one per cluster.
        """
        if anomaly_df.empty:
            return pd.DataFrame(columns=DecisionRecord.__dataclass_fields__.keys())

        records: List[DecisionRecord] = []
        for cid in sorted(anomaly_df["cluster"].unique()):
            subset = anomaly_df[anomaly_df["cluster"] == cid]
            rec    = self._classify_cluster(subset, qresults)
            records.append(rec)

        decisions = pd.DataFrame([vars(r) for r in records])

        # Print triage summary
        print(f"  {'Cluster':>7}  {'Lat':>7}  {'Lon':>7}  "
              f"{'Score':>6}  {'Priority':<10}  Rationale")
        print("  " + "-"*85)
        for _, row in decisions.iterrows():
            print(f"  {int(row.cluster_id):>7}  "
                  f"{row.center_lat:>7.2f}  "
                  f"{row.center_lon:>7.2f}  "
                  f"{row.anomaly_score:>6.2f}  "
                  f"{row.priority_label:<10}  "
                  f"{row.rationale[:55]}")
        print()

        return decisions