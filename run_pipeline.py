"""
QOMAD: Quantum-Inspired Orbital Magnetic Anomaly Detector
Main pipeline orchestrator
"""

import argparse
import numpy as np
from datetime import datetime

from swarm_data import generate_swarm_pass
from anomaly_ml import AnomalyDetector
from quantum_sim import QuantumMagnetometer
from onboard_logic import OnboardDecisionLogic
from globe_viz import render_globe


def run_pipeline(date: str, duration_min: int, quantum_model: str):
    print(f"\n{'='*60}")
    print("  QOMAD — Quantum-Inspired Orbital Magnetic Anomaly Detector")
    print(f"{'='*60}\n")

    # ── 1. Swarm Data ──────────────────────────────────────────────
    print(f"[QOMAD] Loading Swarm Alpha pass: {date} | duration={duration_min} min")
    swarm = generate_swarm_pass(date, duration_min)
    print(f"[QOMAD] Core field subtracted (CHAOS-7 r=13). "
          f"Residual σ = {swarm['residual'].std():.2f} nT\n")

    # ── 2. ML Anomaly Detection ────────────────────────────────────
    print("[QOMAD] Running ML anomaly pipeline...")
    detector = AnomalyDetector()
    anomalies = detector.detect(swarm)
    n_clusters = anomalies['cluster'].nunique()
    max_score  = anomalies['score'].max()
    print(f"[QOMAD] {n_clusters} anomaly cluster(s) detected (max score={max_score:.2f})\n")

    # ── 3. Quantum Magnetometry Simulation ────────────────────────
    print(f"[QOMAD] Simulating quantum magnetometer ({quantum_model})...")
    qmag = QuantumMagnetometer(model=quantum_model)
    qresults = qmag.simulate(swarm['residual'].values)
    print(f"[QOMAD] Quantum SNR gain: +{qresults['snr_gain_db']:.1f} dB vs fluxgate baseline")
    print(f"[QOMAD] Sensitivity: {qresults['sensitivity_fT']:.1f} fT/√Hz  "
          f"(classical: {qresults['classical_fT']:.0f} fT/√Hz)\n")

    # ── 4. Onboard Decision Logic ─────────────────────────────────
    print("[QOMAD] Running onboard decision logic...")
    logic = OnboardDecisionLogic()
    decisions = logic.evaluate(anomalies, qresults)
    n_flagged = decisions['downlink_priority'].sum()
    print(f"[QOMAD] {n_flagged} candidate(s) flagged for priority downlink\n")

    # ── 5. Globe Visualization ─────────────────────────────────────
    outfile = f"output/globe_{date.replace('-','')}.html"
    print(f"[QOMAD] Rendering 3D globe → {outfile}")
    render_globe(swarm, anomalies, decisions, outfile)
    print(f"[QOMAD] Done. Open {outfile} in your browser.\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QOMAD Pipeline")
    parser.add_argument("--date",          default="2023-06-01")
    parser.add_argument("--duration",      type=int, default=282,
                        help="Pass duration in minutes (default=282 = 3 full orbits)")
    parser.add_argument("--quantum-model", default="nv-center",
                        choices=["nv-center", "atomic-rb", "squid"])
    args = parser.parse_args()
    run_pipeline(args.date, args.duration, args.quantum_model)