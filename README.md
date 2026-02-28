# ⚛️ Quantum-Inspired Orbital Magnetic Anomaly Detector (QOMAD)

> Simulating next-generation geomagnetic sensing by fusing ESA Swarm satellite data, machine learning anomaly detection, and quantum magnetometry physics, all in Python.

---

## Description

QOMAD is a Python simulation framework that models the sensitivity advantages of quantum magnetometers in low-Earth orbit. It combines synthetic Swarm geomagnetic field data (with VirES API drop-in support) with an ML anomaly pipeline, then layers a QuTiP-based quantum noise model to demonstrate how NV-diamond and atomic magnetometers outperform classical sensors for crustal anomaly detection. Onboard decision logic autonomously flags anomalies for downlink priority across four tiers (CRITICAL / HIGH / MEDIUM / NOMINAL). Inspired by ESA's SBQuantum partnership, DARPA's TQS program, and SandboxAQ's applied quantum sensing roadmap.

---

## Key Features

- **🌍 Swarm Data Integration**: Simulates ESA Swarm Alpha polar LEO groundtracks with realistic CHAOS-7 core-field subtraction, isolating lithospheric residuals (±60 nT range)
- **🤖 ML Anomaly Detection**: Isolation Forest with sliding-window statistical features (gradient, curvature, rolling σ) detects crustal anomaly signatures from multi-orbit passes
- **⚛️ Quantum Magnetometry Simulation**: Analytical NV-centre, Rb-87 SERF, and SQUID models quantify SNR gains (~38 dB) from quantum projection-noise limits versus fluxgate baselines; QuTiP Lindblad solver used when installed
- **🛰️ Onboard Decision Logic**: Four-tier rule-based classifier (CRITICAL → NOMINAL) with known anomaly catalog cross-reference and novelty detection for autonomous downlink prioritization
- **🌐 Interactive Globe Visualization**: Plotly globe renders multi-orbit groundtracks, field residual heatmap, anomaly cluster markers, priority downlink flags, and known anomaly reference rings with live projection switcher (Globe / Natural Earth / Mercator / Polar)
- **📊 Sensitivity Benchmarking**: Side-by-side classical vs. quantum sensitivity comparison across NV-centre, atomic-Rb, and SQUID architectures

---

## Stack

| Layer | Tools |
|---|---|
| Data | Synthetic Swarm / VirES API (drop-in), CHAOS-7, NumPy, pandas |
| ML | scikit-learn (Isolation Forest), SciPy |
| Quantum Sim | QuTiP (optional), analytical NV/Rb/SQUID models |
| Visualization | Plotly (interactive HTML globe) |
| Orchestration | Python 3.11, argparse |

---

## Setup

```bash
git clone https://github.com/yourhandle/qomad.git
cd qomad
pip install -r requirements.txt
python run_pipeline.py --date 2023-06-01 --quantum-model nv-center
```

Open `output/globe_20230601.html` in your browser when complete. No server required — the globe is a self-contained HTML file.

---

## CLI Options

```
--date            ISO date for the simulated pass  (default: 2023-06-01)
--duration        Pass duration in minutes          (default: 282 = 3 full orbits)
--quantum-model   Sensor model                      (nv-center | atomic-rb | squid)
```

---

## Example Output

```
[QOMAD] Loading Swarm Alpha pass: 2023-06-01 | duration=282 min
[QOMAD] Core field subtracted (CHAOS-7 r=13). Residual σ = 5.19 nT

[QOMAD] ML pipeline: 1 anomaly cluster detected (max score=1.00)
[QOMAD] Quantum SNR gain (NV-centre): +38.5 dB vs fluxgate baseline
[QOMAD] Sensitivity: 1189 fT/√Hz  (classical: 100,000 fT/√Hz)

  Cluster   Lat     Lon    Score  Priority   Rationale
        1   6.12   19.77   1.00  CRITICAL   Novel location (3.4° from Bangui), quantum-confirmed

[QOMAD] 1 candidate flagged for priority downlink
[QOMAD] Globe saved → output/globe_20230601.html
```

---

## References & Inspiration

- [ESA × SBQuantum — Quantum Magnetometers in Orbit](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/ESA_and_SBQuantum)
- [DARPA Tiniest Quantum Sensors (TQS) Program](https://www.darpa.mil/program/tiniest-quantum-sensors)
- [SandboxAQ — Applied Quantum Sensing](https://www.sandboxaq.com/solutions/quantum-sensing)
- [ESA Swarm Mission & VirES Data Portal](https://vires.services)
- [CHAOS-7 Geomagnetic Field Model](https://www.space.dtu.dk/english/research/projects/project-descriptions/chaos)


## 🤝 Contributing
- Contributions are very welcome! If you'd like to collaborate on this project, feel free to:
- Fork the repository and submit a Pull Request
- Open an issue if you find a bug or have a feature idea
- Suggest improvements to the agent pipeline, UI, or documentation
- Share the project with others who might find it useful
- Whether it's a small fix, a new feature, or a completely new idea, all contributions are appreciated.
- Let's build something great together! 🚀


## MIT License
```
Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
---

*Part of an applied quantum sensing & space systems portfolio. Built to explore the intersection of orbital remote sensing, quantum physics simulation, and autonomous edge intelligence.*
