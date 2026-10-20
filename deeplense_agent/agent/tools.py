"""
Agent Tools and Session Context for the DeepLense Simulation Workflow.
"""

from typing import Optional, Dict, Any, List
import uuid
import numpy as np

from deeplense_agent.models.parameters import (
    SimulationParameters,
    ModelConfiguration,
    SubstructureType,
    SubstructureConfig,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
    InstrumentSpecification,
)
from deeplense_agent.models.responses import (
    SimulationPlan,
    ClarificationQuestion,
    SimulationExecutionOutput,
)
from deeplense_agent.simulation.core import (
    DeepLenseSimulationEngine,
    SimulationResultBundle,
)
from deeplense_agent.simulation.instruments import get_instrument_spec


class SimulationSessionContext:
    """Stateful context maintaining active simulation plans and generated results."""
    def __init__(self):
        self.active_plan: Optional[SimulationPlan] = None
        self.last_result: Optional[SimulationResultBundle] = None
        self.plan_history: Dict[str, SimulationPlan] = {}
        self.result_history: Dict[str, SimulationResultBundle] = {}

    def store_plan(self, plan: SimulationPlan) -> None:
        self.active_plan = plan
        self.plan_history[plan.plan_id] = plan

    def store_result(self, plan_id: str, bundle: SimulationResultBundle) -> None:
        self.last_result = bundle
        self.result_history[plan_id] = bundle


def plan_simulation(
    ctx: SimulationSessionContext,
    substructure_type: str = "no_sub",
    model_configuration: str = "Model_II",
    z_lens: Optional[float] = None,
    z_source: Optional[float] = None,
    num_images: int = 1,
    num_subhalos: Optional[int] = None,
    axion_mass_eV: Optional[float] = None,
    custom_resolution_pix: Optional[int] = None,
    require_clarification_on_defaults: bool = True,
) -> SimulationPlan:
    """
    Analyzes requested simulation parameters, applies astrophysical validation,
    and returns a structured SimulationPlan with any HITL clarification questions needed.
    """
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"
    clarifications: List[ClarificationQuestion] = []

    # 1. Parse Substructure Type
    try:
        sub_type = SubstructureType(substructure_type.lower())
    except ValueError:
        sub_type = SubstructureType.NO_SUB
        clarifications.append(
            ClarificationQuestion(
                parameter_name="substructure_type",
                current_or_default_value=sub_type.value,
                suggested_options=["no_sub", "cdm", "axion"],
                question_text=f"Unknown substructure '{substructure_type}'. Defaulting to 'no_sub'. Which dark matter model would you prefer?",
                scientific_rationale="DeepLense investigates dark matter substructure discrimination between smooth lenses, CDM subhalos, and axion vortices.",
            )
        )

    # 2. Parse Model Configuration
    try:
        m_cfg = ModelConfiguration(model_configuration)
    except ValueError:
        m_cfg = ModelConfiguration.MODEL_II
        clarifications.append(
            ClarificationQuestion(
                parameter_name="model_configuration",
                current_or_default_value=m_cfg.value,
                suggested_options=["Model_I", "Model_II", "Model_III", "Model_IV"],
                question_text=f"Selected model '{model_configuration}' unrecognized. Defaulted to Euclid VIS ({m_cfg.value}).",
                scientific_rationale="Different survey characteristics impact resolution and sensitivity: Model_I is Ground-based, Model_II is Euclid VIS, Model_III is HST.",
            )
        )

    # 3. Handle Redshifts & HITL Clarification
    default_zl = 0.5
    default_zs = 1.5

    if z_lens is None:
        z_lens = default_zl
        if require_clarification_on_defaults:
            clarifications.append(
                ClarificationQuestion(
                    parameter_name="z_lens",
                    current_or_default_value=default_zl,
                    suggested_options=["0.3", "0.5", "0.8"],
                    question_text=f"Lens redshift was not specified. I have set z_lens = {default_zl}. Would you like to use this or another value?",
                    scientific_rationale="Deflector distance governs the angular diameter scale and Einstein radius calculation.",
                )
            )

    if z_source is None:
        z_source = default_zs
        if require_clarification_on_defaults:
            clarifications.append(
                ClarificationQuestion(
                    parameter_name="z_source",
                    current_or_default_value=default_zs,
                    suggested_options=["1.0", "1.5", "2.0", "3.0"],
                    question_text=f"Source galaxy redshift was not specified. I have set z_source = {default_zs}. Would you like to confirm this?",
                    scientific_rationale="Source distance determines the optical path and lensing geometry magnification.",
                )
            )

    # 4. Configure Substructure Parameters
    sub_config = SubstructureConfig(substructure_type=sub_type)
    if sub_type == SubstructureType.CDM:
        cdm = CDMSubstructureParameters()
        if num_subhalos is not None:
            cdm.n_sub = int(np.clip(num_subhalos, 1, 200))
        sub_config.cdm_params = cdm

    elif sub_type == SubstructureType.AXION:
        axion = AxionSubstructureParameters()
        if axion_mass_eV is not None:
            axion.axion_mass_eV = float(axion_mass_eV)
        sub_config.axion_params = axion

    # 5. Check Instrument Override
    inst_override: Optional[InstrumentSpecification] = None
    if custom_resolution_pix is not None:
        base_inst = get_instrument_spec(m_cfg)
        inst_override = InstrumentSpecification(
            name=f"{base_inst.name}_Custom{custom_resolution_pix}",
            num_pix=custom_resolution_pix,
            delta_pix=base_inst.delta_pix,
            psf_type=base_inst.psf_type,
            fwhm=base_inst.fwhm,
            exposure_time=base_inst.exposure_time,
            background_rms=base_inst.background_rms,
        )

    # 6. Build and Validate Parameters
    params = SimulationParameters(
        model_configuration=m_cfg,
        substructure=sub_config,
        z_lens=z_lens,
        z_source=z_source,
        num_images=num_images,
        instrument_override=inst_override,
    )

    base_inst = inst_override or get_instrument_spec(m_cfg)
    summary = (
        f"Configured {num_images} image(s) using {m_cfg.value} ({base_inst.name}, {base_inst.num_pix}x{base_inst.num_pix} px). "
        f"Substructure: {sub_type.value}. Redshifts: z_lens={z_lens}, z_source={z_source}."
    )

    plan = SimulationPlan(
        plan_id=plan_id,
        summary=summary,
        parameters=params,
        clarifications=clarifications,
        requires_user_confirmation=len(clarifications) > 0,
        estimated_compute_time_sec=round(0.2 * num_images, 2),
    )

    ctx.store_plan(plan)
    return plan


def refine_simulation_parameters(
    ctx: SimulationSessionContext,
    plan_id: Optional[str] = None,
    z_lens: Optional[float] = None,
    z_source: Optional[float] = None,
    substructure_type: Optional[str] = None,
    model_configuration: Optional[str] = None,
    num_images: Optional[int] = None,
) -> SimulationPlan:
    """
    Refines parameters on an existing simulation plan following user HITL feedback.
    """
    plan = ctx.plan_history.get(plan_id) if plan_id else ctx.active_plan
    if plan is None:
        raise ValueError("No active simulation plan found to refine. Please call plan_simulation first.")

    p_dict = plan.parameters.model_dump()

    if z_lens is not None:
        p_dict["z_lens"] = z_lens
    if z_source is not None:
        p_dict["z_source"] = z_source
    if model_configuration is not None:
        p_dict["model_configuration"] = ModelConfiguration(model_configuration)
    if num_images is not None:
        p_dict["num_images"] = num_images

    if substructure_type is not None:
        st = SubstructureType(substructure_type.lower())
        p_dict["substructure"]["substructure_type"] = st

    new_params = SimulationParameters(**p_dict)
    inst = new_params.instrument_override or get_instrument_spec(new_params.model_configuration)

    updated_summary = (
        f"Refined configuration for {new_params.num_images} image(s) with {new_params.model_configuration.value} "
        f"({inst.name}, {inst.num_pix}x{inst.num_pix} px). Substructure: {new_params.substructure.substructure_type.value}. "
        f"Redshifts: z_lens={new_params.z_lens}, z_source={new_params.z_source}."
    )

    refined_plan = SimulationPlan(
        plan_id=plan.plan_id,
        summary=updated_summary,
        parameters=new_params,
        clarifications=[],
        requires_user_confirmation=False,  # confirmed by refinement
        estimated_compute_time_sec=round(0.2 * new_params.num_images, 2),
    )

    ctx.store_plan(refined_plan)
    return refined_plan


def execute_deeplense_simulation(
    ctx: SimulationSessionContext,
    plan_id: Optional[str] = None,
) -> SimulationExecutionOutput:
    """
    Executes the DeepLense simulation pipeline using the active or specified plan.
    Stores numerical arrays in session context and returns validated structured metadata.
    """
    plan = ctx.plan_history.get(plan_id) if plan_id else ctx.active_plan
    if plan is None:
        raise ValueError("No simulation plan found to execute. Create a plan first via plan_simulation.")

    engine = DeepLenseSimulationEngine(plan.parameters)
    bundle = engine.run(params=plan.parameters, plan_id=plan.plan_id)
    ctx.store_result(plan.plan_id, bundle)

    return bundle.output_metadata


def inspect_simulation_quality(
    ctx: SimulationSessionContext,
    plan_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes astrophysical quality diagnostics on the latest simulation.
    """
    bundle = ctx.result_history.get(plan_id) if plan_id else ctx.last_result
    if bundle is None:
        raise ValueError("No simulation output available to inspect. Execute a simulation first.")

    meta = bundle.output_metadata.images_metadata[0]
    img = bundle.primary_image

    return {
        "plan_id": bundle.output_metadata.plan_id,
        "model": meta.model_configuration.value,
        "substructure": meta.substructure_type.value,
        "resolution": f"{meta.dimensions[0]}x{meta.dimensions[1]}",
        "estimated_snr": meta.estimated_snr,
        "einstein_radius_arcsec": meta.einstein_radius_arcsec,
        "subhalo_components": meta.subhalo_count,
        "peak_flux": meta.max_intensity,
        "background_noise_rms": bundle.parameters.instrument_override.background_rms if bundle.parameters.instrument_override else get_instrument_spec(meta.model_configuration).background_rms,
        "quality_assessment": "EXCELLENT" if meta.estimated_snr > 15 else "LOW_SNR",
    }
