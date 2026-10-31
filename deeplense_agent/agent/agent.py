"""
Pydantic AI Agent instantiation and tool registration for DeepLense.
"""

from typing import Optional, Dict, Any, Union
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry, DeferredToolRequests
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

AgentOutput = Union[str, DeferredToolRequests]


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
    """
    if model is None:
        model = TestModel()

    agent: Agent[SimulationSessionContext, AgentOutput] = Agent(
        model=model,
        system_prompt=DEEPLENSE_SYSTEM_PROMPT,
        deps_type=SimulationSessionContext,
        output_type=[str, DeferredToolRequests],
        retries=3,
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

    return agent
