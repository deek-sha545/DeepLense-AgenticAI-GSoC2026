"""
Instrument and observation configuration profiles for DeepLense Models I through IV.
"""

from typing import Dict, Any
from deeplense_agent.models.parameters import ModelConfiguration, InstrumentSpecification


def get_instrument_spec(model_config: ModelConfiguration) -> InstrumentSpecification:
    """
    Returns the baseline instrument specification for a given DeepLense Model configuration.
    
    Model I:
        Ground-based benchmark (150x150 pixels, pixel size 0.05", Gaussian PSF FWHM=0.087", SNR ~ 25)
    Model II:
        Space-based Euclid VIS survey approximation (64x64 pixels, pixel size 0.1", high sensitivity)
    Model III:
        Space-based Hubble Space Telescope (HST) WFC3/ACS approximation (64x64 pixels, pixel size 0.05")
    Model IV:
        High-fidelity Euclid survey with extended complex galaxy structure (64x64 pixels, pixel size 0.1")
    """
    if model_config == ModelConfiguration.MODEL_I:
        return InstrumentSpecification(
            name="Ground_Benchmark",
            num_pix=150,
            delta_pix=0.05,
            psf_type="GAUSSIAN",
            fwhm=0.087,
            exposure_time=2500.0,
            background_rms=0.01,
        )
    elif model_config == ModelConfiguration.MODEL_II:
        return InstrumentSpecification(
            name="Euclid_VIS",
            num_pix=64,
            delta_pix=0.1,
            psf_type="GAUSSIAN",
            fwhm=0.16,
            exposure_time=565.0 * 4,  # standard Euclid 4-dither co-add
            background_rms=0.005,
        )
    elif model_config == ModelConfiguration.MODEL_III:
        return InstrumentSpecification(
            name="HST_WFC3",
            num_pix=64,
            delta_pix=0.05,
            psf_type="GAUSSIAN",
            fwhm=0.09,
            exposure_time=2000.0,
            background_rms=0.003,
        )
    elif model_config == ModelConfiguration.MODEL_IV:
        return InstrumentSpecification(
            name="Euclid_HighFidelity",
            num_pix=64,
            delta_pix=0.1,
            psf_type="GAUSSIAN",
            fwhm=0.16,
            exposure_time=2260.0,
            background_rms=0.005,
        )
    else:
        # Default fallback
        return InstrumentSpecification(
            name="Generic_Optical",
            num_pix=64,
            delta_pix=0.05,
            psf_type="GAUSSIAN",
            fwhm=0.087,
            exposure_time=2000.0,
            background_rms=0.01,
        )
