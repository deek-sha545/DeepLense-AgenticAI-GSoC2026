"""
Pydantic AI Agent instantiation and tool registration for DeepLense.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry, DeferredToolRequests
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, SystemPromptPart, UserPromptPart
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
    AgentReply,
    SimulationPlan,
    SimulationExecutionOutput,
)

AgentOutput = Union[AgentReply, DeferredToolRequests]

# About 4 characters per token: keeps the history near 4k tokens, so with the system
# prompt and tool schemas a request stays inside free-tier per-request limits (Groq: 8k)
HISTORY_CHAR_BUDGET = 16_000


def _size(message: ModelMessage) -> int:
    return len(ModelMessagesTypeAdapter.dump_json([message]))


def trim_history(messages: List[ModelMessage]) -> List[ModelMessage]:
    """
    Keep the system prompt and the most recent turns that fit HISTORY_CHAR_BUDGET.
    Cuts happen only where a user turn starts, so a tool call is never separated from its
    result, and the current turn is always kept whole. The session state (active plan,
    last result) lives in the deps, not the history, so dropping old turns loses no facts.
    """
    turn_starts = [
        i for i, m in enumerate(messages)
        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)
    ]
    if len(turn_starts) < 2:
        return messages
    cut, used = turn_starts[-1], sum(_size(m) for m in messages[turn_starts[-1]:])
    for start, end in zip(reversed(turn_starts[:-1]), reversed(turn_starts[1:])):
        used += sum(_size(m) for m in messages[start:end])
        if used > HISTORY_CHAR_BUDGET:
            break
        cut = start
    if cut == turn_starts[0]:
        return messages
    system = [p for p in messages[0].parts if isinstance(p, SystemPromptPart)] if isinstance(messages[0], ModelRequest) else []
    first = messages[cut]
    return [ModelRequest(parts=[*system, *first.parts], instructions=first.instructions)] + messages[cut + 1:]


def explain_error(e: Exception) -> str:
    """Compact, model-readable description of a validation failure."""
    if isinstance(e, ValidationError):
        return "; ".join(
            f"{'.'.join(str(x) for x in err['loc']) or 'parameters'}: {err['msg'].removeprefix('Value error, ')} (got {err.get('input')!r})"
            if err["loc"] else err["msg"].removeprefix("Value error, ")
            for err in e.errors()
        )
    return str(e)


def create_pydantic_ai_agent(model: Any = None) -> Agent[SimulationSessionContext, AgentOutput]:
    """
    Creates and configures the Pydantic AI Agent for DeepLense with registered tool functions.

    If model is None, defaults to TestModel() for deterministic offline execution.
    Can also accept any pydantic-ai model string, e.g. 'anthropic:claude-sonnet-5-5',
    'openai:gpt-4o-mini', 'google-gla:gemini-2.5-flash'.

    Human-in-the-loop is enforced by the framework rather than the prompt:
    `tool_execute_simulation` requires approval, so a run that tries to execute
    ends with `DeferredToolRequests` and only continues once the caller passes
    back `DeferredToolResults` with the user's decision.

    The final answer of each turn is a structured `AgentReply`, checked against the
    session state before it is returned; inconsistent replies go back to the LLM.
    """
    if model is None:
        model = TestModel()

    agent: Agent[SimulationSessionContext, AgentOutput] = Agent(
        model=model,
        system_prompt=DEEPLENSE_SYSTEM_PROMPT,
        deps_type=SimulationSessionContext,
        output_type=[AgentReply, DeferredToolRequests],
        retries=3,
        capabilities=[ProcessHistory(trim_history)],
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
        subhalo_mass_slope: Optional[float] = None,
        axion_mass_eV: Optional[float] = None,
        lens_mass_solar: Optional[float] = None,
        velocity_dispersion_kms: Optional[float] = None,
        einstein_radius_arcsec: Optional[float] = None,
        source_magnitude: Optional[float] = None,
        include_lens_light: Optional[bool] = None,
        num_pix: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> SimulationPlan:
        """
        Create a new validated simulation plan from the user's request. Pass only values the
        user stated; omitted redshifts get DeepLense defaults and come back as clarifications.

        Args:
            substructure_type: 'no_sub' (smooth), 'cdm' (subhalos) or 'axion' (vortex).
            model_configuration: 'Model_I' (generic 150px), 'Model_II' (Euclid), 'Model_III' (HST), 'Model_IV' (Euclid, real-galaxy source).
            z_lens: Lens redshift.
            z_source: Source redshift, must exceed z_lens.
            num_images: Images to generate, 1-50.
            num_subhalos: CDM subhalo count, 1-200 (default 25).
            subhalo_mass_slope: CDM mass function slope in dN/dm ~ m^-|slope|, magnitude 0.1 to 2.5 (default 0.9; CDM theory ~1.9).
            axion_mass_eV: Axion mass in eV, 1e-25 to 1e-19; unset draws 1e-24 to 1e-22 per image.
            lens_mass_solar: Main halo mass in M_sun (default 1e12), sets theta_E.
            velocity_dispersion_kms: Lens velocity dispersion in km/s, sets theta_E instead of mass.
            einstein_radius_arcsec: Set theta_E directly in arcsec.
            source_magnitude: Source apparent magnitude for Models II-IV (default 20; fainter is larger).
            include_lens_light: Add deflector galaxy light (default only for Model IV).
            num_pix: Override image size in pixels, 32-512.
            random_seed: Seed for an exactly reproducible run.
        """
        try:
            return plan_simulation(
                ctx.deps,
                substructure_type=substructure_type,
                model_configuration=model_configuration,
                z_lens=z_lens,
                z_source=z_source,
                num_images=num_images,
                num_subhalos=num_subhalos,
                subhalo_mass_slope=subhalo_mass_slope,
                axion_mass_eV=axion_mass_eV,
                lens_mass_solar=lens_mass_solar,
                velocity_dispersion_kms=velocity_dispersion_kms,
                einstein_radius_arcsec=einstein_radius_arcsec,
                source_magnitude=source_magnitude,
                include_lens_light=include_lens_light,
                num_pix=num_pix,
                random_seed=random_seed,
            )
        except ValueError as e:
            raise ModelRetry(f"Invalid simulation parameters: {explain_error(e)}")

    @agent.tool
    def tool_refine_parameters(
        ctx: RunContext[SimulationSessionContext],
        substructure_type: Optional[str] = None,
        model_configuration: Optional[str] = None,
        z_lens: Optional[float] = None,
        z_source: Optional[float] = None,
        num_images: Optional[int] = None,
        num_subhalos: Optional[int] = None,
        subhalo_mass_slope: Optional[float] = None,
        axion_mass_eV: Optional[float] = None,
        lens_mass_solar: Optional[float] = None,
        velocity_dispersion_kms: Optional[float] = None,
        einstein_radius_arcsec: Optional[float] = None,
        source_magnitude: Optional[float] = None,
        include_lens_light: Optional[bool] = None,
        num_pix: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> SimulationPlan:
        """
        Change fields of the current plan. Pass only the fields the user wants changed;
        everything else is kept. Arguments mean the same as in tool_plan_simulation.
        """
        try:
            return refine_simulation_parameters(
                ctx.deps,
                substructure_type=substructure_type,
                model_configuration=model_configuration,
                z_lens=z_lens,
                z_source=z_source,
                num_images=num_images,
                num_subhalos=num_subhalos,
                subhalo_mass_slope=subhalo_mass_slope,
                axion_mass_eV=axion_mass_eV,
                lens_mass_solar=lens_mass_solar,
                velocity_dispersion_kms=velocity_dispersion_kms,
                einstein_radius_arcsec=einstein_radius_arcsec,
                source_magnitude=source_magnitude,
                include_lens_light=include_lens_light,
                num_pix=num_pix,
                random_seed=random_seed,
            )
        except ValueError as e:
            raise ModelRetry(f"Could not refine plan: {explain_error(e)}")

    @agent.tool(requires_approval=True)
    def tool_execute_simulation(ctx: RunContext[SimulationSessionContext]) -> SimulationExecutionOutput:
        """
        Run the current plan. The user sees the plan and must approve before it runs.
        Returns per-image metadata: Einstein radius, arc SNR, substructure count, seed.
        """
        try:
            return execute_deeplense_simulation(ctx.deps)
        except ValueError as e:
            raise ModelRetry(str(e))

    @agent.tool
    def tool_inspect_diagnostics(ctx: RunContext[SimulationSessionContext]) -> Dict[str, Any]:
        """Quality diagnostics for the latest run: arc SNR across images, ring size in pixels, and suggestions."""
        try:
            return inspect_simulation_quality(ctx.deps)
        except ValueError as e:
            raise ModelRetry(str(e))

    @agent.output_validator
    def validate_reply(ctx: RunContext[SimulationSessionContext], output: AgentOutput) -> AgentOutput:
        """Reject replies that contradict the session: results without a run, or invented defaults."""
        if not isinstance(output, AgentReply):
            return output
        plan = ctx.deps.active_plan
        if output.intent == "results" and ctx.deps.last_result is None:
            raise ModelRetry("No simulation has run yet, so intent cannot be 'results'. Nothing runs until the user approves.")
        if output.intent == "plan" and plan is None:
            raise ModelRetry("There is no plan yet. Call tool_plan_simulation before replying with intent 'plan'.")
        flagged = {c.parameter_name for c in plan.clarifications} if plan is not None else set()
        invented = [name for name in output.defaulted_parameters if name not in flagged]
        if invented:
            raise ModelRetry(
                f"defaulted_parameters lists {invented}, but the plan only flagged {sorted(flagged) or 'nothing'}. "
                "List only parameters from the plan's clarifications."
            )
        return output

    return agent
