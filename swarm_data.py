"""
swarm_data.py — Synthetic ESA Swarm geomagnetic pass generator

Simulates Swarm Alpha L2 magnetic residuals after CHAOS-7 core-field
subtraction. In production, replace with viresclient API calls.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Approximate Swarm Alpha orbital parameters (LEO ~460 km)
SWARM_ALTITUDE_KM  = 460.0
SWARM_INCLINATION  = 87.4   # degrees
SWARM_PERIOD_MIN   = 94.0   # minutes


def _great_circle_track(start_lat: float, start_lon: float,
                         n_points: int, dt_sec: float) -> tuple:
    """
    Generate a realistic polar LEO groundtrack.
    Uses a sinusoidal latitude model (correct for near-polar orbits)
    with Earth rotation baked into the longitude drift.
    """
    orbital_period_s  = SWARM_PERIOD_MIN * 60
    angular_velocity  = 2 * np.pi / orbital_period_s   # rad/s
    earth_rotation    = 2 * np.pi / 86400              # rad/s (sidereal)

    t   = np.arange(n_points) * dt_sec
    inc = np.radians(SWARM_INCLINATION)

    # True anomaly (radians)
    nu  = angular_velocity * t

    # Geocentric latitude from orbital mechanics
    lat = np.degrees(np.arcsin(np.sin(inc) * np.sin(nu)))

    # Longitude: ascending node + in-plane angle - Earth rotation
    raan_drift  = np.radians(start_lon)
    lon_rad     = raan_drift + np.arctan2(np.cos(inc) * np.sin(nu), np.cos(nu)) \
                  - earth_rotation * t
    lon = (np.degrees(lon_rad) + 180) % 360 - 180

    return lat, lon


def _chaos7_core_field(lat: np.ndarray, lon: np.ndarray, alt_km: float) -> np.ndarray:
    """Simplified CHAOS-7 core field model (dipole + low-degree terms)."""
    theta = np.radians(90 - lat)          # colatitude
    phi   = np.radians(lon)
    r     = 6371 + alt_km                 # geocentric radius km

    # Dominant dipole (g10 = -29404 nT at Earth surface)
    g10 = -29404.0
    B_r = 2 * g10 * (6371 / r)**3 * np.cos(theta)

    # Add quadrupole perturbation
    g20 = -1450.0
    B_r += g20 * (6371 / r)**4 * (3 * np.cos(theta)**2 - 1) / 2

    # Low-amplitude longitudinal variation
    B_r += 800 * np.sin(2 * phi) * np.sin(theta)

    return B_r


def _lithospheric_anomalies(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Synthetic lithospheric/crustal anomalies.
    Mimics localized magnetic bodies (e.g., Bangui anomaly, Kursk anomaly).
    """
    signal = np.zeros(len(lat))

    anomaly_sources = [
        # (center_lat, center_lon, amplitude_nT, width_deg)
        (5.0,   23.0,  120.0, 3.0),   # Bangui-like (Central Africa)
        (52.0,  37.5,   80.0, 2.5),   # Kursk-like (Russia)
        (-30.0, 25.0,   60.0, 4.0),   # South African craton
        (35.0, 135.0,   45.0, 2.0),   # Japanese arc
        (-55.0, -65.0,  35.0, 3.5),   # Antarctic Peninsula
    ]

    for clat, clon, amp, width in anomaly_sources:
        dist = np.sqrt((lat - clat)**2 + (lon - clon)**2)
        signal += amp * np.exp(-(dist**2) / (2 * width**2))

    return signal


def generate_swarm_pass(date: str, duration_min: int,
                         dt_sec: float = 4.0) -> pd.DataFrame:
    """
    Generate a synthetic Swarm orbital pass with magnetic residuals.

    Parameters
    ----------
    date        : ISO date string  e.g. '2023-06-01'
    duration_min: length of pass in minutes
    dt_sec      : sample cadence in seconds (Swarm L2 = 1 Hz; 4 s for speed)

    Returns
    -------
    pd.DataFrame with columns:
        time, lat, lon, alt_km, B_core, B_lithosphere,
        measurement_nT, residual
    """
    np.random.seed(int(datetime.strptime(date, "%Y-%m-%d").timestamp()) % 2**31)

    n_points   = int(duration_min * 60 / dt_sec)
    start_lat  = np.random.uniform(-70, 70)
    start_lon  = np.random.uniform(-180, 180)

    timestamps = [datetime.strptime(date, "%Y-%m-%d") + timedelta(seconds=i * dt_sec)
                  for i in range(n_points)]

    lat, lon = _great_circle_track(start_lat, start_lon, n_points, dt_sec)

    B_core        = _chaos7_core_field(lat, lon, SWARM_ALTITUDE_KM)
    B_litho       = _lithospheric_anomalies(lat, lon)

    # Instrument noise (fluxgate: ~100 pT rms, mapped to nT)
    noise         = np.random.normal(0, 0.1, n_points)

    # Space weather / magnetospheric contamination
    external      = 3.0 * np.sin(2 * np.pi * np.arange(n_points) / (n_points / 2))

    measurement   = B_core + B_litho + external + noise
    residual      = measurement - B_core          # core-subtracted residual

    df = pd.DataFrame({
        "time":           timestamps,
        "lat":            lat,
        "lon":            lon,
        "alt_km":         SWARM_ALTITUDE_KM,
        "B_core_nT":      B_core,
        "B_litho_nT":     B_litho,
        "measurement_nT": measurement,
        "residual":       residual,
        "external_nT":    external,
        "noise_nT":       noise,
    })

    return df