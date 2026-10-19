"""Stage labels, probability-column order and canonical model fields."""

from enum import Enum


class FlightPhase(str, Enum):
    """Four stages; UNKNOWN is a rejected prediction, not a training label."""

    GROUND = "GROUND"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"
    UNKNOWN = "UNKNOWN"


class FiveStageFlightPhase(str, Enum):
    """Five stages, distinguishing level flight (LVL) from cruise (CR)."""

    GROUND = "GROUND"
    CLIMB = "CLIMB"
    LEVEL = "LVL"
    CRUISE = "CR"
    DESCENT = "DESCENT"
    UNKNOWN = "UNKNOWN"


ANNOTATABLE_PHASES = tuple(phase for phase in FlightPhase if phase is not FlightPhase.UNKNOWN)
FIVE_STAGE_ANNOTATABLE_PHASES = tuple(
    phase for phase in FiveStageFlightPhase if phase is not FiveStageFlightPhase.UNKNOWN
)
FOUR_STAGE_PHASES = tuple(phase.value for phase in ANNOTATABLE_PHASES)
FIVE_STAGE_PHASES = tuple(phase.value for phase in FIVE_STAGE_ANNOTATABLE_PHASES)

BASE_FEATURE_COLUMNS = ("altitude", "vertrate", "velocity")
TRAJECTORY_ID_COLUMN = "flight_id"
TIMESTAMP_COLUMN = "timestamp"
