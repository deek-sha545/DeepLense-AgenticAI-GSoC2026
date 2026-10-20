"""Simulation package for DeepLense Agent."""

from deeplense_agent.simulation.core import (
    DeepLenseSimulationEngine,
    SimulationResultBundle,
)
from deeplense_agent.simulation.instruments import get_instrument_spec
from deeplense_agent.simulation.substructures import (
    build_lens_model,
    compute_einstein_radius,
)

__all__ = [
    "DeepLenseSimulationEngine",
    "SimulationResultBundle",
    "get_instrument_spec",
    "build_lens_model",
    "compute_einstein_radius",
]
