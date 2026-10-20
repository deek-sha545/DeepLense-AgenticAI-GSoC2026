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
    assert bundle.output_metadata.images_metadata[0].estimated_snr > 10.0


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
