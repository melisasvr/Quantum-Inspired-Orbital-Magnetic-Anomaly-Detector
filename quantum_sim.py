"""
quantum_sim.py — Quantum magnetometer simulation via QuTiP

Models three sensor architectures:
  nv-center  : NV-diamond spin-1 system (SBQuantum / DeBeers approach)
  atomic-rb  : Rubidium-87 SERF atomic magnetometer
  squid      : Superconducting quantum interference device

Solves the Lindblad master equation to compute:
  - Steady-state spin polarisation
  - Projection-noise-limited sensitivity
  - SNR gain over classical fluxgate baseline

References:
  Degen et al., Rev. Mod. Phys. 89, 035002 (2017)
  Rondin et al., Rep. Prog. Phys. 77, 056503 (2014)
"""

import numpy as np

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("[quantum_sim] WARNING: qutip not installed. Using analytical approximation.")


# ── Physical constants ────────────────────────────────────────────────────

HBAR       = 1.0545718e-34   # J·s
MU_B       = 9.2740100e-24   # J/T  (Bohr magneton)
GAMMA_NV   = 28.035e9        # Hz/T  NV gyromagnetic ratio
GAMMA_RB87 = 6.9954e9        # Hz/T  Rb-87 ground state
GAMMA_E    = 1.7609e11       # rad/(s·T)


# ── NV-centre model ──────────────────────────────────────────────────────

class NVCenterModel:
    """
    Spin-1 NV-centre magnetometer in the electronic ground state (³A₂).

    Hamiltonian:  H = D*Sz² + γ_NV * B * Sz
    D = 2.87 GHz zero-field splitting.
    """

    D_ZFS_HZ = 2.87e9   # zero-field splitting (Hz)

    def __init__(self, T2_us: float = 1.0, T1_us: float = 6.0,
                 n_qubits: int = 1):
        self.T2  = T2_us  * 1e-6   # coherence time (s)
        self.T1  = T1_us  * 1e-6   # relaxation time (s)
        self.n_q = n_qubits

    def sensitivity_fT_rtHz(self, B_field_T: float = 1e-9,
                             tau_s: float = 1.0) -> float:
        """
        Projection-noise-limited sensitivity for a single NV centre.
        η = 1 / (γ_NV * C * √(N * τ * T2))
        where C = contrast (~0.03), N = number of NV spins.
        """
        C    = 0.03           # ODMR contrast
        N    = 1e12           # spin ensemble density (per cm³ × vol)
        eta  = 1.0 / (GAMMA_NV * C * np.sqrt(N * tau_s * self.T2))
        return eta * 1e15    # convert T/√Hz → fT/√Hz

    def build_hamiltonian(self, B_field_T: float = 0.0):
        """Return spin-1 NV Hamiltonian as QuTiP Qobj."""
        if not QUTIP_AVAILABLE:
            return None
        Sz = qt.jmat(1, 'z')
        H  = 2 * np.pi * self.D_ZFS_HZ * Sz * Sz + \
             2 * np.pi * GAMMA_NV * B_field_T * Sz
        return H

    def collapse_operators(self):
        """Lindblad collapse operators for T1, T2 decay."""
        if not QUTIP_AVAILABLE:
            return []
        Sz  = qt.jmat(1, 'z')
        Sp  = qt.jmat(1, '+')
        Sm  = qt.jmat(1, '-')
        c_ops = [
            np.sqrt(1 / self.T1) * Sm,                    # spin relaxation
            np.sqrt(1 / (2 * self.T2)) * Sz,              # pure dephasing
        ]
        return c_ops

    def steady_state_polarisation(self, B_field_T: float = 1e-9) -> float:
        """Compute ⟨Sz⟩ at steady state under optical pumping."""
        if not QUTIP_AVAILABLE:
            # Analytical approximation
            omega_L = GAMMA_NV * B_field_T
            return np.tanh(omega_L * self.T2) * 0.85
        H    = self.build_hamiltonian(B_field_T)
        c_ops = self.collapse_operators()
        rho_ss = qt.steadystate(H, c_ops)
        Sz = qt.jmat(1, 'z')
        return float(qt.expect(Sz, rho_ss).real)


# ── Atomic Rb-87 model ────────────────────────────────────────────────────

class AtomicRbModel:
    """
    SERF (Spin-Exchange Relaxation-Free) Rb-87 magnetometer.
    Operates near zero field; extremely high sensitivity regime.
    """

    def __init__(self, T2_ms: float = 100.0):
        self.T2 = T2_ms * 1e-3

    def sensitivity_fT_rtHz(self, n_atoms: float = 1e14,
                             tau_s: float = 1.0) -> float:
        """
        Shot-noise limit: η ≈ 1 / (γ * √(n * τ * T2))
        """
        eta = 1.0 / (GAMMA_RB87 * np.sqrt(n_atoms * tau_s * self.T2))
        return eta * 1e15

    def snr_vs_classical(self) -> float:
        """SNR gain over fluxgate (100 pT/√Hz baseline)."""
        classical_fT = 100_000.0   # 100 pT = 100,000 fT
        quantum_fT   = self.sensitivity_fT_rtHz()
        return 20 * np.log10(classical_fT / max(quantum_fT, 1e-3))


# ── SQUID model ───────────────────────────────────────────────────────────

class SQUIDModel:
    """Simplified DC-SQUID sensitivity model."""

    def sensitivity_fT_rtHz(self) -> float:
        """Typical HTS SQUID: ~10 fT/√Hz at room temperature."""
        return 10.0

    def snr_vs_classical(self) -> float:
        classical_fT = 100_000.0
        return 20 * np.log10(classical_fT / self.sensitivity_fT_rtHz())


# ── Main interface ────────────────────────────────────────────────────────

class QuantumMagnetometer:
    """
    Unified quantum magnetometer simulator.

    Wraps NV-centre, atomic-Rb, or SQUID models and applies
    simulated quantum noise filtering to a residual field time series.
    """

    CLASSICAL_SENSITIVITY_FT = 100_000.0   # fluxgate: ~100 pT/√Hz

    MODELS = {
        "nv-center":  NVCenterModel,
        "atomic-rb":  AtomicRbModel,
        "squid":      SQUIDModel,
    }

    def __init__(self, model: str = "nv-center"):
        if model not in self.MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(self.MODELS)}")
        self.model_name = model
        self._sensor    = self.MODELS[model]()

    def _quantum_filter(self, signal: np.ndarray,
                        snr_gain_linear: float) -> np.ndarray:
        """
        Apply simulated quantum noise reduction.
        Models reduced white-noise floor proportional to SNR gain.
        """
        noise_scale = 1.0 / np.sqrt(snr_gain_linear)
        filtered    = np.zeros_like(signal)

        # Simple exponential smoothing (mimics projection-noise averaging)
        alpha = 1.0 - noise_scale
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]

        # Residual white noise at quantum level
        q_noise = np.random.normal(0, noise_scale * signal.std() * 0.05,
                                   len(signal))
        return filtered + q_noise

    def simulate(self, residual: np.ndarray,
                 integration_time_s: float = 1.0) -> dict:
        """
        Simulate quantum magnetometer output on residual field array.

        Returns dict with:
            sensitivity_fT  : quantum sensitivity in fT/√Hz
            classical_fT    : classical baseline in fT/√Hz
            snr_gain_db     : SNR improvement in dB
            filtered_signal : noise-reduced residual (nT)
            polarisation    : (NV only) mean spin polarisation
        """
        # Get sensitivity
        if hasattr(self._sensor, 'sensitivity_fT_rtHz'):
            q_sens = self._sensor.sensitivity_fT_rtHz()
        else:
            q_sens = 10.0

        snr_gain_db  = 20 * np.log10(
            self.CLASSICAL_SENSITIVITY_FT / max(q_sens, 1e-3))
        snr_linear   = 10 ** (snr_gain_db / 20)

        filtered = self._quantum_filter(residual, snr_linear)

        # NV spin polarisation across the pass
        polarisation = None
        if self.model_name == "nv-center":
            nv = self._sensor
            B_values = residual * 1e-9   # nT → T
            pol = [nv.steady_state_polarisation(float(b)) for b in
                   B_values[::max(1, len(B_values)//50)]]   # subsample
            polarisation = np.mean(np.abs(pol))

        return {
            "sensitivity_fT":  q_sens,
            "classical_fT":    self.CLASSICAL_SENSITIVITY_FT,
            "snr_gain_db":     snr_gain_db,
            "snr_linear":      snr_linear,
            "filtered_signal": filtered,
            "polarisation":    polarisation,
            "model":           self.model_name,
        }

    def sensitivity_comparison(self) -> dict:
        """Return sensitivity comparison table across all models."""
        results = {}
        for name, cls in self.MODELS.items():
            s = cls()
            sens = s.sensitivity_fT_rtHz() if hasattr(s, 'sensitivity_fT_rtHz') else 10.0
            gain = 20 * np.log10(self.CLASSICAL_SENSITIVITY_FT / max(sens, 1e-3))
            results[name] = {"sensitivity_fT": sens, "snr_gain_db": gain}
        results["fluxgate"] = {
            "sensitivity_fT": self.CLASSICAL_SENSITIVITY_FT,
            "snr_gain_db": 0.0,
        }
        return results