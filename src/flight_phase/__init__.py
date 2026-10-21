"""Core package for ADS-B flight-phase recognition."""

from flight_phase.data import load_trajectory, normalize_trajectory

__version__ = "0.1.0"

__all__ = ["__version__", "load_trajectory", "normalize_trajectory"]
