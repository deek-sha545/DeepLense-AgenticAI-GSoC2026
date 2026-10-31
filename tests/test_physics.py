"""
Physics sanity checks for the simulation engine and the agent's parameter tools.
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from deeplense_agent.models import (
    SimulationParameters,
    SubstructureConfig,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
    LensMacroParameters,
)
from deeplense_agent.simulation import DeepLenseSimulationEngine, compute_einstein_radius
from deeplense_agent.simulation.substructures import (
    build_lens_model,
    isothermal_einstein_radius,
    axion_mass_to_de_broglie_arcsec,
)
from deeplense_agent.agent.tools import (
    SimulationSessionContext,
    apply_parameter_updates,
    plan_simulation,
    refine_simulation_parameters,
    execute_deeplense_simulation,
    inspect_simulation_quality,
)

COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)


def run(**kwargs):
    return DeepLenseSimulationEngine(SimulationParameters(**kwargs)).run()


def test_point_mass_einstein_radius_scales_as_sqrt_mass():
    t1 = compute_einstein_radius(1e12, 0.5, 1.0, COSMO)
    t4 = compute_einstein_radius(4e12, 0.5, 1.0, COSMO)
    assert t4 == pytest.approx(2 * t1, rel=1e-6)


def test_isothermal_einstein_radius():
    # theta_E = 4 pi (sigma/c)^2 D_ls/D_s; ~1" for sigma = 250 km/s at z = 0.5 -> 1.0
    theta = isothermal_einstein_radius(250, 0.5, 1.0, COSMO)
    d_ratio = COSMO.angular_diameter_distance(0.5, 1.0) / COSMO.angular_diameter_distance(1.0)
    expected = 4 * np.pi * (250 / 299792.458) ** 2 * float(d_ratio) * 206264.806
    assert theta == pytest.approx(expected, rel=1e-6)
    assert isothermal_einstein_radius(500, 0.5, 1.0, COSMO) == pytest.approx(4 * theta, rel=1e-6)


def test_cdm_count_and_placement_follow_deeplense():
    cdm = CDMSubstructureParameters(n_sub=40, r_min=0.3, r_max=1.8)
    r = build_lens_model(LensMacroParameters(), 0.5, 1.0, COSMO, "cdm", cdm_params=cdm, rng=np.random.RandomState(0))
    subs = [k for m, k in zip(r.model_list, r.kwargs_list) if m == "POINT_MASS"]
    radii = np.hypot([k["center_x"] for k in subs], [k["center_y"] for k in subs])
    assert len(subs) == r.substructure_count == 40
    assert radii.min() >= 0.3 and radii.max() <= 1.8
    assert 40 * 1e6 <= r.substructure_mass <= 40 * 1e10


def test_axion_vortex_length_shrinks_with_mass():
    assert axion_mass_to_de_broglie_arcsec(1e-23) == pytest.approx(10 * axion_mass_to_de_broglie_arcsec(1e-22))
    axion = AxionSubstructureParameters(axion_mass_eV=1e-23)
    r = build_lens_model(LensMacroParameters(), 0.5, 1.0, COSMO, "axion", axion_params=axion, rng=np.random.RandomState(0))
    xs = np.array([k["center_x"] for m, k in zip(r.model_list, r.kwargs_list) if m == "POINT_MASS"])
    ys = np.array([k["center_y"] for m, k in zip(r.model_list, r.kwargs_list) if m == "POINT_MASS"])
    length = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    assert length == pytest.approx(2 * axion_mass_to_de_broglie_arcsec(1e-23), rel=1e-6)
    assert r.axion_mass_eV == 1e-23


def test_seeded_runs_are_reproducible_and_unseeded_runs_record_their_seed():
    a = run(substructure=SubstructureConfig(substructure_type="cdm"), random_seed=3)
    b = run(substructure=SubstructureConfig(substructure_type="cdm"), random_seed=3)
    c = run(substructure=SubstructureConfig(substructure_type="cdm"), random_seed=4)
    assert np.array_equal(a.images_observed, b.images_observed)
    assert not np.array_equal(a.images_observed, c.images_observed)

    free = run(substructure=SubstructureConfig(substructure_type="cdm"))
    replay = run(substructure=SubstructureConfig(substructure_type="cdm"), random_seed=free.output_metadata.random_seed)
    assert np.array_equal(free.images_observed, replay.images_observed)


@pytest.mark.parametrize("model", ["Model_I", "Model_II", "Model_III", "Model_IV"])
def test_noise_matches_reported_background(model):
    b = run(model_configuration=model, random_seed=1, source_params={"center_x": 0.0, "center_y": 0.0}, include_lens_light=False)
    meta = b.output_metadata.images_metadata[0]
    corner = (b.images_observed[0] - b.images_noiseless[0])[:12, :12]
    assert corner.std() == pytest.approx(meta.background_rms, rel=0.25)


def test_brighter_source_has_higher_snr():
    faint = run(model_configuration="Model_II", source_params={"magnitude": 24}, random_seed=1)
    bright = run(model_configuration="Model_II", source_params={"magnitude": 20}, random_seed=1)
    assert bright.output_metadata.images_metadata[0].snr > 5 * faint.output_metadata.images_metadata[0].snr


def test_convergence_map_shows_point_mass_substructure():
    smooth = run(random_seed=2)
    cdm = run(substructure=SubstructureConfig(substructure_type="cdm"), random_seed=2)
    excess = (cdm.kappa_maps[0] - smooth.kappa_maps[0]).sum()
    assert excess > 0  # point masses used to vanish from the kappa map entirely


def test_model_iv_uses_galaxy_source_lens_light_and_fixed_einstein_radius(tmp_path):
    b = run(model_configuration="Model_IV", random_seed=5)
    meta = b.output_metadata.images_metadata[0]
    assert meta.source_type == "procedural_galaxy"
    assert meta.lens_light is True
    assert meta.einstein_radius_arcsec == 1.5

    img = np.zeros((40, 40))
    img[15:25, 10:30] = 1.0
    path = tmp_path / "galaxy.npy"
    np.save(path, img)
    b = run(model_configuration="Model_IV", source_params={"galaxy_image_path": str(path)}, random_seed=5)
    assert b.output_metadata.images_metadata[0].source_type == "galaxy_image"

    heavy = run(model_configuration="Model_IV", lens_params={"mass_solar": 4e12}, random_seed=5)
    assert heavy.output_metadata.images_metadata[0].einstein_radius_arcsec > 3.0


def test_refinement_keeps_model_specific_defaults():
    ctx = SimulationSessionContext()
    plan_simulation(ctx, model_configuration="Model_IV", z_lens=0.5, z_source=1.0)
    refined = refine_simulation_parameters(ctx, num_images=2)
    lens = refined.parameters.lens_params
    assert "e1" not in lens.model_fields_set and "mass_solar" not in lens.model_fields_set
    execute_deeplense_simulation(ctx)
    assert ctx.last_result.output_metadata.images_metadata[0].einstein_radius_arcsec == 1.5


def test_einstein_radius_inputs_are_mutually_exclusive():
    p = apply_parameter_updates(einstein_radius_arcsec=1.2)
    p = apply_parameter_updates(p, velocity_dispersion_kms=230)
    assert p.lens_params.theta_E is None and p.lens_params.velocity_dispersion_kms == 230
    p = apply_parameter_updates(p, lens_mass_solar=5e11)
    assert p.lens_params.velocity_dispersion_kms is None and p.lens_params.mass_solar == 5e11


def test_switching_substructure_drops_stale_parameters():
    p = apply_parameter_updates(substructure_type="cdm", num_subhalos=60)
    p = apply_parameter_updates(p, substructure_type="axion", axion_mass_eV=1e-23)
    assert p.substructure.cdm_params is None
    assert p.substructure.axion_params.axion_mass_eV == 1e-23


def test_quality_inspection_flags_faint_arcs():
    ctx = SimulationSessionContext()
    plan_simulation(ctx, z_lens=0.5, z_source=1.0, source_magnitude=27, random_seed=1)
    execute_deeplense_simulation(ctx)
    report = inspect_simulation_quality(ctx)
    assert report["quality_assessment"] == "LOW_SNR"
    assert report["suggestions"]


def test_subhalo_slope_of_one_samples_log_uniform_masses():
    import numpy as np
    from astropy.cosmology import FlatLambdaCDM
    from deeplense_agent.models import CDMSubstructureParameters, LensMacroParameters
    from deeplense_agent.simulation.substructures import build_lens_model

    cdm = CDMSubstructureParameters(n_sub=200, beta=-1.0)
    real = build_lens_model(LensMacroParameters(), 0.5, 1.0, FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05),
                            "cdm", cdm_params=cdm, rng=np.random.RandomState(1))
    assert real.substructure_count == 200
    assert np.isfinite(real.substructure_mass) and 200 * 1e6 < real.substructure_mass < 200 * 1e10
