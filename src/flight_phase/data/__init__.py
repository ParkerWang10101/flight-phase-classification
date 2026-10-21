"""Canonical ADS-B-equivalent trajectory interfaces."""

from flight_phase.data.trajectory import normalize_trajectory
from flight_phase.data.io import load_trajectory

__all__ = ["load_trajectory", "normalize_trajectory"]
