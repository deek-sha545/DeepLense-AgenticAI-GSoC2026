"""
Unit Tests for DeepLense Simulation Pipeline.
"""

import pytest
import numpy as np
from deeplense_agent.models import (
    SimulationParameters,
    ModelConfiguration,
    SubstructureType,
    SubstructureConfig,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
)
from deeplense_agent.simulation import (
    DeepLenseSimulationEngine,
    compute_einstein_radius,
)
from astropy.cosmology import FlatLambdaCDM


def test_einstein_radius_calculation():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    theta_e = compute_einstein_radius(1e12, 0.5, 1.5, cosmo)
    # Typical galaxy-scale Einstein radius should be ~1.2 - 1.8 arcsec
    assert 1.0 < theta_e < 2.0


def test_model_i_ground_benchmark():
    params = SimulationParameters(
        model_configuration=ModelConfiguration.MODEL_I,
        substructure=SubstructureConfig(substructure_type=SubstructureType.NO_SUB),
        num_images=1,
        random_seed=42,
    )
    engine = DeepLenseSimulationEngine(params)
    bundle = engine.run()

    # Model I should have 150x150 dimensions
    assert bundle.images_observed.shape == (1, 150, 150)
    assert bundle.kappa_maps.shape == (1, 150, 150)
    assert bundle.source_planes.shape == (1, 150, 150)
    assert bundle.output_metadata.success is True
    assert bundle.output_metadata.images_metadata[0].snr > 10.0


def test_model_ii_euclid_survey():
    params = SimulationParameters(
        model_configuration=ModelConfiguration.MODEL_II,
        substructure=SubstructureConfig(substructure_type=SubstructureType.CDM),
        num_images=1,
        random_seed=42,
    )
    engine = DeepLenseSimulationEngine(params)
    bundle = engine.run()

    # Model II should have 64x64 dimensions
    assert bundle.images_observed.shape == (1, 64, 64)
    meta = bundle.output_metadata.images_metadata[0]
    assert meta.model_configuration == ModelConfiguration.MODEL_II
    assert meta.substructure_type == SubstructureType.CDM
    assert meta.subhalo_count > 0


def test_model_iii_hst_survey():
    params = SimulationParameters(
        model_configuration=ModelConfiguration.MODEL_III,
        substructure=SubstructureConfig(substructure_type=SubstructureType.AXION),
        num_images=1,
        random_seed=42,
    )
    engine = DeepLenseSimulationEngine(params)
    bundle = engine.run()

    assert bundle.images_observed.shape == (1, 64, 64)
    meta = bundle.output_metadata.images_metadata[0]
    assert meta.model_configuration == ModelConfiguration.MODEL_III
    assert meta.substructure_type == SubstructureType.AXION
    assert meta.subhalo_count == 100


@pytest.mark.parametrize("model", [ModelConfiguration.MODEL_I, ModelConfiguration.MODEL_II])
def test_same_seed_pairs_share_source_and_noise_across_substructure(model):
    """A smooth and a CDM run with one seed must differ only through the lens."""
    def run(sub):
        p = SimulationParameters(model_configuration=model, substructure=dict(substructure_type=sub),
                                 num_images=2, random_seed=2024)
        return DeepLenseSimulationEngine(p).run(p)

    smooth, cdm = run(SubstructureType.NO_SUB), run(SubstructureType.CDM)
    for a, b in zip(smooth.output_metadata.images_metadata, cdm.output_metadata.images_metadata):
        assert a.source_center_arcsec == b.source_center_arcsec
        assert a.exposure_time_s == b.exposure_time_s
    assert cdm.output_metadata.images_metadata[0].subhalo_count == 25

    # Shared noise realization: the paired difference of noise residuals is far below one noise draw
    for i in range(2):
        res_s = smooth.images_observed[i] - smooth.images_noiseless[i]
        res_c = cdm.images_observed[i] - cdm.images_noiseless[i]
        assert np.std(res_s - res_c) < 0.2 * np.std(res_s)
        assert not np.allclose(smooth.images_noiseless[i], cdm.images_noiseless[i])


def test_shot_noise_keeps_poisson_statistics():
    from deeplense_agent.simulation.core import add_noise
    flux, exposure = np.full((200, 200), 2.0), 50.0  # 100 counts per pixel
    noisy = add_noise(flux, exposure, 0.0, np.random.RandomState(0))
    counts = noisy * exposure
    assert abs(counts.mean() - 100) < 0.5
    assert abs(counts.var() - 100) < 5
    assert np.allclose(counts, np.round(counts))
