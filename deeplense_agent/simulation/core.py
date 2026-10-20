"""
Core Simulation Engine wrapping Lenstronomy for DeepLenseSim.
Simulates strong gravitational lensing with full instrument and noise characteristics.
"""

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

from deeplense_agent.models.parameters import (
    SimulationParameters,
    ModelConfiguration,
    SubstructureType,
)
from deeplense_agent.models.responses import (
    SingleImageMetadata,
    SimulationExecutionOutput,
)
from deeplense_agent.simulation.instruments import get_instrument_spec
from deeplense_agent.simulation.substructures import (
    build_lens_model,
    compute_einstein_radius,
)


class SimulationResultBundle:
    """In-memory bundle containing both visual/numerical arrays and structured metadata."""
    def __init__(
        self,
        images_observed: np.ndarray,      # shape (N, H, W)
        images_noiseless: np.ndarray,     # shape (N, H, W)
        source_planes: np.ndarray,        # shape (N, H, W)
        kappa_maps: np.ndarray,           # shape (N, H, W)
        output_metadata: SimulationExecutionOutput,
        parameters: SimulationParameters,
    ):
        self.images_observed = images_observed
        self.images_noiseless = images_noiseless
        self.source_planes = source_planes
        self.kappa_maps = kappa_maps
        self.output_metadata = output_metadata
        self.parameters = parameters

    @property
    def primary_image(self) -> np.ndarray:
        """First observed mock image."""
        return self.images_observed[0]

    @property
    def primary_kappa(self) -> np.ndarray:
        """First dark matter convergence map."""
        return self.kappa_maps[0]

    @property
    def primary_source(self) -> np.ndarray:
        """First source plane unlensed galaxy image."""
        return self.source_planes[0]


class DeepLenseSimulationEngine:
    """
    Simulation engine implementing DeepLense Models I, II, III, and IV.
    """

    def __init__(self, parameters: Optional[SimulationParameters] = None):
        self.parameters = parameters or SimulationParameters()
        self._cosmo = FlatLambdaCDM(
            H0=self.parameters.cosmology.H0,
            Om0=self.parameters.cosmology.Om0,
            Ob0=self.parameters.cosmology.Ob0,
        )

    def run(self, params: Optional[SimulationParameters] = None, plan_id: str = "plan_default") -> SimulationResultBundle:
        """
        Executes the simulation pipeline for the provided parameters.
        Returns a SimulationResultBundle containing arrays and validated metadata.
        """
        start_time = time.time()
        p = params or self.parameters
        rng = np.random.RandomState(p.random_seed)

        # 1. Determine Instrument Spec
        if p.instrument_override:
            inst = p.instrument_override
        else:
            inst = get_instrument_spec(p.model_configuration)

        num_pix = inst.num_pix
        delta_pix = inst.delta_pix

        # 2. Build Coordinate Grid
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, m_pix2coord, _ = util.make_grid_with_coordtransform(
            num_pix=num_pix,
            delta_pix=delta_pix,
            center_ra=0.0,
            center_dec=0.0,
            subgrid_res=1,
            inverse=False,
        )

        kwargs_data = {
            "background_rms": inst.background_rms,
            "exposure_time": inst.exposure_time,
            "ra_at_xy_0": ra_at_xy_0,
            "dec_at_xy_0": dec_at_xy_0,
            "transform_pix2angle": m_pix2coord,
            "image_data": np.zeros((num_pix, num_pix)),
        }
        data_class = ImageData(**kwargs_data)

        # 3. Setup PSF
        kwargs_psf = {
            "psf_type": inst.psf_type,
            "fwhm": inst.fwhm,
            "pixel_size": delta_pix,
            "truncation": 4,
        }
        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}

        # Containers for batch outputs
        observed_list: List[np.ndarray] = []
        noiseless_list: List[np.ndarray] = []
        source_plane_list: List[np.ndarray] = []
        kappa_list: List[np.ndarray] = []
        image_metadata_list: List[SingleImageMetadata] = []

        # 4. Generate Images
        for idx in range(p.num_images):
            # Build Lens Model with requested substructure
            sub_type_val = p.substructure.substructure_type.value
            lens_model_list, kwargs_lens, lens_redshifts, sub_count = build_lens_model(
                lens_params=p.lens_params,
                z_lens=p.z_lens,
                z_source=p.z_source,
                cosmo_config=p.cosmology,
                substructure_type=sub_type_val,
                cdm_params=p.substructure.cdm_params,
                axion_params=p.substructure.axion_params,
                rng=rng,
            )
            lens_model_class = LensModel(lens_model_list=lens_model_list)

            # Build Source Light Model
            # Allow slight position jitter for batch variety if num_images > 1
            src_x = p.source_params.center_x + (rng.uniform(-0.05, 0.05) if idx > 0 else 0.0)
            src_y = p.source_params.center_y + (rng.uniform(-0.05, 0.05) if idx > 0 else 0.0)

            # Intensity amplitude handling
            if p.model_configuration in (ModelConfiguration.MODEL_II, ModelConfiguration.MODEL_III, ModelConfiguration.MODEL_IV):
                # Magnitude based amplitude scaling
                flux_amp = 10.0 ** (-0.4 * (p.source_params.magnitude - 22.5))
            else:
                flux_amp = p.source_params.amp if p.source_params.amp is not None else 20.0

            source_model_list = ["SERSIC_ELLIPSE"]
            kwargs_source = [
                {
                    "amp": float(flux_amp),
                    "R_sersic": p.source_params.R_sersic,
                    "n_sersic": p.source_params.n_sersic,
                    "e1": p.source_params.e1,
                    "e2": p.source_params.e2,
                    "center_x": src_x,
                    "center_y": src_y,
                }
            ]
            source_model_class = LightModel(light_model_list=source_model_list)

            # Assemble ImageModel
            im_sim = ImageModel(
                data_class=data_class,
                psf_class=psf_class,
                lens_model_class=lens_model_class,
                source_model_class=source_model_class,
                kwargs_numerics=kwargs_numerics,
            )

            # Compute noiseless lensed image
            model_flux = im_sim.image(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                kwargs_lens_light=None,
                kwargs_ps=None,
            )

            # Apply realistic instrument noise
            # 1. Poisson shot noise from source photon counts
            poisson_noise = image_util.add_poisson(model_flux, exp_time=inst.exposure_time)
            # 2. Gaussian background detector read noise
            bkg_noise = image_util.add_background(model_flux, sigma_bkd=inst.background_rms)
            sim_real = (model_flux + poisson_noise + bkg_noise).clip(min=0.0)

            # Generate Source Plane image (unlensed)
            x_grid, y_grid = util.make_grid(num_pix=num_pix, delta_pix=delta_pix)
            src_plane_flux = source_model_class.surface_brightness(x_grid, y_grid, kwargs_source)
            src_plane = util.array2image(src_plane_flux)

            # Compute Dark Matter Convergence map (kappa)
            kappa_raw = lens_model_class.kappa(x_grid, y_grid, kwargs_lens)
            kappa_map = util.array2image(kappa_raw)

            # Calculate Einstein Radius
            theta_e = compute_einstein_radius(
                p.lens_params.mass_solar, p.z_lens, p.z_source, self._cosmo
            )

            # Estimate SNR
            peak_val = float(np.max(sim_real))
            bkg_val = float(inst.background_rms)
            snr_est = float(peak_val / bkg_val) if bkg_val > 0 else 0.0

            # Image Metadata
            img_meta = SingleImageMetadata(
                image_index=idx,
                model_configuration=p.model_configuration,
                substructure_type=p.substructure.substructure_type,
                dimensions=(num_pix, num_pix),
                pixel_scale_arcsec=delta_pix,
                z_lens=p.z_lens,
                z_source=p.z_source,
                einstein_radius_arcsec=round(theta_e, 4),
                min_intensity=round(float(np.min(sim_real)), 4),
                max_intensity=round(peak_val, 4),
                mean_intensity=round(float(np.mean(sim_real)), 4),
                estimated_snr=round(snr_est, 2),
                subhalo_count=sub_count,
                notes=[
                    f"Simulated via DeepLenseSim engine using {inst.name} specs.",
                    f"Substructure type: {p.substructure.substructure_type.value}.",
                ],
            )

            observed_list.append(sim_real)
            noiseless_list.append(model_flux)
            source_plane_list.append(src_plane)
            kappa_list.append(kappa_map)
            image_metadata_list.append(img_meta)

        elapsed = time.time() - start_time

        # Compile execution summary
        summary = (
            f"Successfully generated {p.num_images} simulation(s) using {p.model_configuration.value} "
            f"({inst.name}) with {p.substructure.substructure_type.value} substructure. "
            f"Image dimensions: {num_pix}x{num_pix} at {delta_pix}\"/pix. "
            f"Redshifts: z_lens={p.z_lens}, z_source={p.z_source} (theta_E={theta_e:.3f}\")."
        )

        exec_output = SimulationExecutionOutput(
            plan_id=plan_id,
            success=True,
            num_images_generated=p.num_images,
            model_configuration=p.model_configuration,
            substructure_type=p.substructure.substructure_type,
            images_metadata=image_metadata_list,
            execution_time_seconds=round(elapsed, 3),
            summary=summary,
        )

        return SimulationResultBundle(
            images_observed=np.array(observed_list),
            images_noiseless=np.array(noiseless_list),
            source_planes=np.array(source_plane_list),
            kappa_maps=np.array(kappa_list),
            output_metadata=exec_output,
            parameters=p,
        )
