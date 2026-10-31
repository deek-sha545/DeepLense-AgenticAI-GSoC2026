"""
Agent Tools and Session Context for the DeepLense Simulation Workflow.
"""

from typing import Optional, Dict, Any, List
import uuid
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from deeplense_agent.models.parameters import (
    SimulationParameters,
    ModelConfiguration,
    SubstructureType,
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
from deeplense_agent.simulation.substructures import main_einstein_radius

DEFAULT_Z_LENS = 0.5
DEFAULT_Z_SOURCE = 1.0
SECONDS_PER_IMAGE = 0.05


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


def apply_parameter_updates(
    base: Optional[SimulationParameters] = None,
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
    galaxy_image_path: Optional[str] = None,
) -> SimulationParameters:
    """
    Returns new validated parameters with the given fields changed. Unset arguments keep
    the base value. Only explicitly set fields are carried over, so model-specific
    defaults (e.g. Model IV's lens) keep applying after a refinement.
    """
    d: Dict[str, Any] = base.model_dump(exclude_unset=True) if base is not None else {}
    sub = d.setdefault("substructure", {})
    lens = d.setdefault("lens_params", {})
    source = d.setdefault("source_params", {})

    if model_configuration is not None:
        d["model_configuration"] = ModelConfiguration(model_configuration)
    if z_lens is not None:
        d["z_lens"] = z_lens
    if z_source is not None:
        d["z_source"] = z_source
    if num_images is not None:
        d["num_images"] = num_images
    if include_lens_light is not None:
        d["include_lens_light"] = include_lens_light
    if num_pix is not None:
        d["num_pix"] = num_pix
    if random_seed is not None:
        d["random_seed"] = random_seed

    if substructure_type is not None:
        new_type = SubstructureType(substructure_type.lower())
        if new_type.value != SubstructureType(sub.get("substructure_type", "no_sub")).value:
            sub.pop("cdm_params", None)
            sub.pop("axion_params", None)
        sub["substructure_type"] = new_type
    if num_subhalos is not None:
        sub.setdefault("cdm_params", {})["n_sub"] = int(np.clip(num_subhalos, 1, 200))
    if subhalo_mass_slope is not None:
        # "slope 1.9" and "slope -1.9" both mean dN/dm ~ m^-1.9 in common usage
        sub.setdefault("cdm_params", {})["beta"] = -abs(subhalo_mass_slope)
    if axion_mass_eV is not None:
        sub.setdefault("axion_params", {})["axion_mass_eV"] = axion_mass_eV

    # The three ways of setting theta_E are mutually exclusive; the latest one wins
    if einstein_radius_arcsec is not None:
        lens.pop("velocity_dispersion_kms", None)
        lens["theta_E"] = einstein_radius_arcsec
    if velocity_dispersion_kms is not None:
        lens.pop("theta_E", None)
        lens["velocity_dispersion_kms"] = velocity_dispersion_kms
    if lens_mass_solar is not None:
        lens.pop("theta_E", None)
        lens.pop("velocity_dispersion_kms", None)
        lens["mass_solar"] = lens_mass_solar

    if source_magnitude is not None:
        source["magnitude"] = source_magnitude
    if galaxy_image_path is not None:
        source["galaxy_image_path"] = galaxy_image_path

    return SimulationParameters(**d)


def parameter_facts(p: SimulationParameters) -> Dict[str, str]:
    """Resolved, human-readable facts about a parameter set (instrument, theta_E, source, ...)."""
    inst = get_instrument_spec(p.model_configuration, p.num_pix)
    sub = p.substructure.substructure_type.value
    if sub == "cdm":
        cdm = p.substructure.cdm_params
        sub = f"cdm ({cdm.n_sub} subhalos, slope {cdm.beta})"
    elif sub == "axion":
        m = p.substructure.axion_params.axion_mass_eV
        sub = f"axion (m_a = {m:.1e} eV)" if m else "axion (m_a drawn from 1e-24 to 1e-22 eV per image)"

    lens = DeepLenseSimulationEngine._effective_lens_params(p)
    cosmo = FlatLambdaCDM(H0=p.cosmology.H0, Om0=p.cosmology.Om0, Ob0=p.cosmology.Ob0)
    theta_e = main_einstein_radius(lens, p.z_lens, p.z_source, cosmo)
    if lens.theta_E is not None and p.lens_params.theta_E is None:
        origin = "Model IV default"
    elif lens.theta_E is not None:
        origin = "set directly"
    elif lens.velocity_dispersion_kms is not None:
        origin = f"from sigma_v = {lens.velocity_dispersion_kms:g} km/s"
    else:
        origin = f"from M = {lens.mass_solar:.1e} M_sun"

    if p.model_configuration == ModelConfiguration.MODEL_IV:
        path = p.source_params.galaxy_image_path
        source = f"galaxy image ({path.rsplit('/', 1)[-1]})" if path else "procedural spiral galaxy"
    else:
        source = f"Sersic, R = {p.source_params.R_sersic}\""
    brightness = f"amp {p.source_params.amp:g}" if inst.flux_units == "amplitude" else f"mag {p.source_params.magnitude:g}"

    return {
        "instrument": f"{p.model_configuration.value} ({inst.name})",
        "grid": f"{inst.num_pix}x{inst.num_pix} px at {inst.delta_pix}\"/px",
        "substructure": sub,
        "redshifts": f"z_lens={p.z_lens}, z_source={p.z_source}",
        "einstein_radius": f"{theta_e:.2f}\" ({origin})",
        "source": f"{source}, {brightness}",
        "lens_light": "on" if p.lens_light_enabled else "off",
        "images": str(p.num_images),
        "seed": str(p.random_seed) if p.random_seed is not None else "random",
    }


def describe_parameters(p: SimulationParameters) -> str:
    """One-paragraph human summary of a parameter set."""
    f = parameter_facts(p)
    return (
        f"{f['images']} image(s), {f['instrument']}, {f['grid']}. Substructure: {f['substructure']}. "
        f"{f['redshifts']}. theta_E = {f['einstein_radius']}. Source: {f['source']}. "
        f"Lens light: {f['lens_light']}. Seed: {f['seed']}."
    )


def plan_simulation(
    ctx: SimulationSessionContext,
    substructure_type: str = "no_sub",
    model_configuration: str = "Model_II",
    z_lens: Optional[float] = None,
    z_source: Optional[float] = None,
    num_images: int = 1,
    require_clarification_on_defaults: bool = True,
    **updates: Any,
) -> SimulationPlan:
    """
    Builds and validates a new SimulationPlan. Omitted redshifts fall back to the
    DeepLense defaults (z_lens = 0.5, z_source = 1.0) and are flagged as clarifications.
    Extra keyword arguments are any of `apply_parameter_updates`'s fields.
    """
    clarifications: List[ClarificationQuestion] = []

    try:
        SubstructureType(substructure_type.lower())
    except ValueError:
        clarifications.append(ClarificationQuestion(
            parameter_name="substructure_type",
            current_or_default_value="no_sub",
            suggested_options=["no_sub", "cdm", "axion"],
            question_text=f"Unknown substructure '{substructure_type}', so I used a smooth lens. Which dark matter model do you want?",
            scientific_rationale="DeepLense classifies lenses by substructure: smooth, CDM subhalos, or axion vortices.",
        ))
        substructure_type = "no_sub"

    try:
        ModelConfiguration(model_configuration)
    except ValueError:
        clarifications.append(ClarificationQuestion(
            parameter_name="model_configuration",
            current_or_default_value="Model_II",
            suggested_options=[m.value for m in ModelConfiguration],
            question_text=f"Unknown configuration '{model_configuration}', so I used Model_II (Euclid VIS).",
            scientific_rationale="The configuration sets resolution, PSF and noise: Model_I generic, II Euclid, III HST, IV Euclid with real galaxy sources.",
        ))
        model_configuration = "Model_II"

    if z_lens is None and require_clarification_on_defaults:
        clarifications.append(ClarificationQuestion(
            parameter_name="z_lens",
            current_or_default_value=DEFAULT_Z_LENS,
            suggested_options=["0.3", "0.5", "0.8"],
            question_text=f"No lens redshift given, so I used the DeepLense default z_lens = {DEFAULT_Z_LENS}.",
            scientific_rationale="The deflector distance sets the angular scale and, with the mass, the Einstein radius.",
        ))
    if z_source is None and require_clarification_on_defaults:
        clarifications.append(ClarificationQuestion(
            parameter_name="z_source",
            current_or_default_value=DEFAULT_Z_SOURCE,
            suggested_options=["1.0", "1.5", "2.0"],
            question_text=f"No source redshift given, so I used the DeepLense default z_source = {DEFAULT_Z_SOURCE}.",
            scientific_rationale="The source distance sets the lensing efficiency D_ls/D_s and so the Einstein radius.",
        ))

    params = apply_parameter_updates(
        substructure_type=substructure_type,
        model_configuration=model_configuration,
        z_lens=z_lens if z_lens is not None else DEFAULT_Z_LENS,
        z_source=z_source if z_source is not None else DEFAULT_Z_SOURCE,
        num_images=num_images,
        **updates,
    )
    plan = SimulationPlan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        summary=describe_parameters(params),
        parameters=params,
        clarifications=clarifications,
        requires_user_confirmation=True,
        estimated_compute_time_sec=round(SECONDS_PER_IMAGE * params.num_images, 2),
    )
    ctx.store_plan(plan)
    return plan


def refine_simulation_parameters(
    ctx: SimulationSessionContext,
    plan_id: Optional[str] = None,
    **updates: Any,
) -> SimulationPlan:
    """
    Applies user changes to an existing plan. Keyword arguments are any of
    `apply_parameter_updates`'s fields; None values are ignored.
    """
    plan = ctx.plan_history.get(plan_id) if plan_id else ctx.active_plan
    if plan is None:
        raise ValueError("No active simulation plan found to refine. Please call plan_simulation first.")

    new_params = apply_parameter_updates(plan.parameters, **updates)
    refined = SimulationPlan(
        plan_id=plan.plan_id,
        summary=describe_parameters(new_params),
        parameters=new_params,
        clarifications=[],
        requires_user_confirmation=True,
        estimated_compute_time_sec=round(SECONDS_PER_IMAGE * new_params.num_images, 2),
    )
    ctx.store_plan(refined)
    return refined


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

    bundle = DeepLenseSimulationEngine(plan.parameters).run(plan_id=plan.plan_id)
    ctx.store_result(plan.plan_id, bundle)
    return bundle.output_metadata


def inspect_simulation_quality(
    ctx: SimulationSessionContext,
    plan_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quality diagnostics over every image of the latest (or given) run, with
    concrete suggestions when the arcs are faint or the lens is poorly sampled.
    """
    bundle = ctx.result_history.get(plan_id) if plan_id else ctx.last_result
    if bundle is None:
        raise ValueError("No simulation output available to inspect. Execute a simulation first.")

    metas = bundle.output_metadata.images_metadata
    snrs = np.array([m.snr for m in metas])
    first = metas[0]
    theta_pix = first.einstein_radius_arcsec / first.pixel_scale_arcsec
    fov_half = first.dimensions[0] * first.pixel_scale_arcsec / 2

    suggestions: List[str] = []
    if snrs.min() < 10:
        suggestions.append("Some arcs have integrated SNR < 10; use a brighter source magnitude or a deeper instrument.")
    if theta_pix < 5:
        suggestions.append("The Einstein ring spans fewer than 5 pixels; substructure will be hard to resolve.")
    if first.einstein_radius_arcsec > 0.8 * fov_half:
        suggestions.append("The Einstein ring nearly fills the field of view; increase num_pix or lower the lens mass.")

    if snrs.min() >= 50:
        quality = "HIGH_SNR"
    elif snrs.min() >= 10:
        quality = "DETECTABLE"
    else:
        quality = "LOW_SNR"

    return {
        "plan_id": bundle.output_metadata.plan_id,
        "random_seed": bundle.output_metadata.random_seed,
        "model": first.model_configuration.value,
        "substructure": first.substructure_type.value,
        "num_images": len(metas),
        "resolution": f"{first.dimensions[0]}x{first.dimensions[1]} at {first.pixel_scale_arcsec}\"/px",
        "einstein_radius_arcsec": first.einstein_radius_arcsec,
        "einstein_radius_pixels": round(theta_pix, 1),
        "arc_snr_min_median_max": [round(float(snrs.min()), 1), round(float(np.median(snrs)), 1), round(float(snrs.max()), 1)],
        "peak_pixel_snr_median": round(float(np.median([m.peak_snr for m in metas])), 1),
        "background_rms": first.background_rms,
        "substructure_components": first.subhalo_count,
        "substructure_mass_msun": first.substructure_mass_msun,
        "quality_assessment": quality,
        "suggestions": suggestions,
    }
