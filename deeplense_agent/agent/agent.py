"""
Pydantic AI Agent instantiation and tool registration for DeepLense.
"""

from typing import Optional, Dict, Any, List
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from deeplense_agent.agent.prompts import DEEPLENSE_SYSTEM_PROMPT
from deeplense_agent.agent.tools import (
    SimulationSessionContext,
    plan_simulation,
    refine_simulation_parameters,
    execute_deeplense_simulation,
    inspect_simulation_quality,
)
from deeplense_agent.models.responses import (
    SimulationPlan,
    SimulationExecutionOutput,
)


def create_pydantic_ai_agent(model: Any = None) -> Agent[SimulationSessionContext, str]:
    """
    Creates and configures the Pydantic AI Agent for DeepLense with registered tool functions.
    
    If model is None, defaults to TestModel() for deterministic offline execution.
    Can also accept 'google-gla:gemini-1.5-flash', 'openai:gpt-4o-mini', etc.
    """
    if model is None:
        model = TestModel()

    agent: Agent[SimulationSessionContext, str] = Agent(
        model=model,
        system_prompt=DEEPLENSE_SYSTEM_PROMPT,
        deps_type=SimulationSessionContext,
    )

    @agent.tool
    def tool_plan_simulation(
        ctx: RunContext[SimulationSessionContext],
        substructure_type: str = "no_sub",
        model_configuration: str = "Model_II",
        z_lens: Optional[float] = None,
        z_source: Optional[float] = None,
        num_images: int = 1,
        num_subhalos: Optional[int] = None,
        custom_resolution_pix: Optional[int] = None,
    ) -> SimulationPlan:
        """
        Analyze simulation request, enforce astrophysics guardrails, and create a SimulationPlan.
        Generates clarification questions if parameters are ambiguous or omitted.
        """
        return plan_simulation(
            ctx=ctx.deps,
            substructure_type=substructure_type,
            model_configuration=model_configuration,
            z_lens=z_lens,
            z_source=z_source,
            num_images=num_images,
            num_subhalos=num_subhalos,
            custom_resolution_pix=custom_resolution_pix,
        )

    @agent.tool
    def tool_refine_parameters(
        ctx: RunContext[SimulationSessionContext],
        plan_id: Optional[str] = None,
        z_lens: Optional[float] = None,
        z_source: Optional[float] = None,
        substructure_type: Optional[str] = None,
        model_configuration: Optional[str] = None,
        num_images: Optional[int] = None,
    ) -> SimulationPlan:
        """Refine parameters on an existing simulation plan based on user feedback."""
        return refine_simulation_parameters(
            ctx=ctx.deps,
            plan_id=plan_id,
            z_lens=z_lens,
            z_source=z_source,
            substructure_type=substructure_type,
            model_configuration=model_configuration,
            num_images=num_images,
        )

    @agent.tool
    def tool_execute_simulation(
        ctx: RunContext[SimulationSessionContext],
        plan_id: Optional[str] = None,
    ) -> SimulationExecutionOutput:
        """
        Execute the DeepLense simulation pipeline with confirmed parameters.
        Generates images, dark matter convergence maps, and returns structured metadata.
        """
        return execute_deeplense_simulation(ctx=ctx.deps, plan_id=plan_id)

    @agent.tool
    def tool_inspect_diagnostics(
        ctx: RunContext[SimulationSessionContext],
        plan_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inspect SNR, substructure count, and image quality metrics of the generated simulation."""
        return inspect_simulation_quality(ctx=ctx.deps, plan_id=plan_id)

    return agent
