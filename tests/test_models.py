"""
Unit Tests for Pydantic Models and Astrophysics Guardrails.
"""

import pytest
from pydantic import ValidationError
from deeplense_agent.models import (
    SimulationParameters,
    ModelConfiguration,
    SubstructureType,
    SubstructureConfig,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
    SimulationPlan,
    ClarificationQuestion,
)


def test_valid_simulation_parameters():
    params = SimulationParameters(
        model_configuration=ModelConfiguration.MODEL_II,
        z_lens=0.5,
        z_source=1.5,
        num_images=2,
    )
    assert params.model_configuration == ModelConfiguration.MODEL_II
    assert params.z_lens == 0.5
    assert params.z_source == 1.5
    assert params.num_images == 2


def test_redshift_ordering_guardrail():
    # z_lens >= z_source violates physics
    with pytest.raises(ValidationError) as exc_info:
        SimulationParameters(
            z_lens=1.5,
            z_source=0.5,
        )
    assert "Lens redshift z_lens" in str(exc_info.value)


def test_cdm_mass_range_guardrail():
    # m_sub_min >= m_sub_max violates physics
    with pytest.raises(ValidationError) as exc_info:
        CDMSubstructureParameters(
            m_sub_min=1e10,
            m_sub_max=1e8,
        )
    assert "m_sub_min" in str(exc_info.value)


def test_substructure_auto_population():
    cfg = SubstructureConfig(substructure_type=SubstructureType.CDM)
    assert cfg.cdm_params is not None
    assert cfg.cdm_params.n_sub == 25

    cfg_axion = SubstructureConfig(substructure_type=SubstructureType.AXION)
    assert cfg_axion.axion_params is not None
    assert cfg_axion.axion_params.axion_mass_eV == 1e-22
