"""
Instrument and observation configuration profiles for DeepLense Models I through IV.

Model I is the DeepLenseSim generic benchmark (amplitude units, fixed background RMS).
Models II-IV use lenstronomy's survey observation configs so that pixel scale, PSF,
sky brightness, read noise and zero point all come from one consistent source.
"""

from typing import Any, Dict, Optional

from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

from deeplense_agent.models.parameters import ModelConfiguration, InstrumentSpecification

SURVEY_NUM_PIX = 64

# Model I: DeepLenseSim lens.py `simple_sim`
MODEL_I_NUM_PIX = 150
MODEL_I_DELTA_PIX = 0.05
MODEL_I_FWHM = 0.087
MODEL_I_BACKGROUND_RMS = 1e-2
MODEL_I_LOG10_EXPOSURE_RANGE = (3.0, 3.5)


def survey_band_config(model_config: ModelConfiguration) -> Optional[Dict[str, Any]]:
    """lenstronomy `kwargs_single_band` for survey models, None for Model I."""
    if model_config in (ModelConfiguration.MODEL_II, ModelConfiguration.MODEL_IV):
        return Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()
    if model_config == ModelConfiguration.MODEL_III:
        return HST(band="WFC3_F160W", psf_type="GAUSSIAN").kwargs_single_band()
    return None


def get_instrument_spec(model_config: ModelConfiguration, num_pix: Optional[int] = None) -> InstrumentSpecification:
    """Resolved instrument description for a model configuration, optionally with a custom image size."""
    band = survey_band_config(model_config)
    if band is None:
        return InstrumentSpecification(
            name="Generic_Optical",
            num_pix=num_pix or MODEL_I_NUM_PIX,
            delta_pix=MODEL_I_DELTA_PIX,
            fwhm=MODEL_I_FWHM,
            exposure_time=10 ** (sum(MODEL_I_LOG10_EXPOSURE_RANGE) / 2),
            background_rms=MODEL_I_BACKGROUND_RMS,
            flux_units="amplitude",
        )
    name = {
        ModelConfiguration.MODEL_II: "Euclid_VIS",
        ModelConfiguration.MODEL_III: "HST_WFC3_F160W",
        ModelConfiguration.MODEL_IV: "Euclid_VIS_RealGalaxy",
    }[model_config]
    return InstrumentSpecification(
        name=name,
        num_pix=num_pix or SURVEY_NUM_PIX,
        delta_pix=band["pixel_scale"],
        fwhm=band["seeing"],
        exposure_time=band["exposure_time"] * band["num_exposures"],
        flux_units="magnitude",
    )
