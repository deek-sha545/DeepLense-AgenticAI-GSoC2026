"""
DeepLense Agentic AI Package.
An agentic workflow wrapping the DeepLenseSim simulation pipeline for strong gravitational lensing.
"""

from deeplense_agent.models import (
    SubstructureType,
    ModelConfiguration,
    SimulationParameters,
    SimulationPlan,
    SingleImageMetadata,
    SimulationExecutionOutput,
)
from deeplense_agent.simulation import (
    DeepLenseSimulationEngine,
    SimulationResultBundle,
)
from deeplense_agent.agent import (
    create_pydantic_ai_agent,
    DeepLenseWorkflowSession,
    WorkflowState,
    AgentTurnResponse,
)
from deeplense_agent.visualization import (
    plot_lensing_diagnostic,
    plot_batch_comparison,
)

__version__ = "0.1.0"

__all__ = [
    "SubstructureType",
    "ModelConfiguration",
    "SimulationParameters",
    "SimulationPlan",
    "SingleImageMetadata",
    "SimulationExecutionOutput",
    "DeepLenseSimulationEngine",
    "SimulationResultBundle",
    "create_pydantic_ai_agent",
    "DeepLenseWorkflowSession",
    "WorkflowState",
    "AgentTurnResponse",
    "plot_lensing_diagnostic",
    "plot_batch_comparison",
]
