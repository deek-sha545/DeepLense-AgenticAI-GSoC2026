"""
Core Simulation Engine wrapping Lenstronomy for DeepLenseSim.
Simulates strong gravitational lensing with full instrument and noise characteristics.

Model I renders in amplitude units with a fixed background RMS, as DeepLenseSim does.
Models II-IV render through lenstronomy's SimAPI with survey observation configs, so
magnitudes, sky background, read noise and PSF are consistent. All randomness (lens
realization, source position, noise) is drawn from one seeded generator, and the seed
is recorded in the output metadata so every run can be reproduced exactly.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.util as util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI

from deeplense_agent.models.parameters import (
    ModelConfiguration,
    SimulationParameters,
)
from deeplense_agent.models.responses import (
    SimulationExecutionOutput,
    SingleImageMetadata,
)
from deeplense_agent.simulation.instruments import (
    MODEL_I_LOG10_EXPOSURE_RANGE,
    get_instrument_spec,
    survey_band_config,
)
from deeplense_agent.simulation.sources import draw_source_galaxy
from deeplense_agent.simulation.substructures import (
    LensModelRealization,
    build_lens_model,
    main_einstein_radius,
)

KWARGS_NUMERICS = {"supersampling_factor": 2, "supersampling_convolution": False}
MODEL_IV_THETA_E = 1.5


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


def pixelized_kappa(
    realization: LensModelRealization,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    num_pix: int,
    delta_pix: float,
) -> np.ndarray:
    """
    Convergence map averaged over pixels. Smooth profiles are evaluated analytically;
    point masses (whose convergence is a delta function) deposit their integrated
    convergence, pi * theta_E^2, into the pixel that contains them.
    """
    smooth = [(m, k) for m, k in zip(realization.model_list, realization.kwargs_list) if m != "POINT_MASS"]
    lens = LensModel([m for m, _ in smooth])
    kappa = util.array2image(lens.kappa(x_grid, y_grid, [k for _, k in smooth]))

    half = num_pix * delta_pix / 2.0
    for m, k in zip(realization.model_list, realization.kwargs_list):
        if m != "POINT_MASS":
            continue
        i = int((k["center_y"] + half) // delta_pix)
        j = int((k["center_x"] + half) // delta_pix)
        if 0 <= i < num_pix and 0 <= j < num_pix:
            kappa[i, j] += np.pi * k["theta_E"] ** 2 / delta_pix ** 2
    return kappa


def add_noise(image: np.ndarray, exposure_time: float, background_rms: float, rng: np.random.RandomState) -> np.ndarray:
    """Poisson shot noise on source counts plus Gaussian background, in counts/s."""
    counts = np.clip(image, 0.0, None) * exposure_time
    shot = rng.poisson(counts) / exposure_time - np.clip(image, 0.0, None)
    background = rng.normal(0.0, background_rms, size=image.shape)
    return image + shot + background


def signal_to_noise(arc: np.ndarray, total: np.ndarray, exposure_time: float, background_rms: float) -> tuple:
    """
    Integrated SNR of the lensed arc over pixels where the arc exceeds the background RMS:
        SNR = sum(arc) / sqrt(sum(var)),  var = sigma_bkg^2 + model / t_exp
    Also returns the best single-pixel SNR.
    """
    variance = background_rms ** 2 + np.clip(total, 0.0, None) / exposure_time
    per_pixel = arc / np.sqrt(variance)
    mask = arc > background_rms
    if not mask.any():
        return 0.0, float(per_pixel.max())
    integrated = float(arc[mask].sum() / np.sqrt(variance[mask].sum()))
    return integrated, float(per_pixel.max())


class DeepLenseSimulationEngine:
    """
    Simulation engine implementing DeepLense Models I, II, III, and IV.
    """

    def __init__(self, parameters: Optional[SimulationParameters] = None):
        self.parameters = parameters or SimulationParameters()

    def run(self, params: Optional[SimulationParameters] = None, plan_id: str = "plan_default") -> SimulationResultBundle:
        """
        Executes the simulation pipeline for the provided parameters.
        Returns a SimulationResultBundle containing arrays and validated metadata.
        """
        start_time = time.time()
        p = params or self.parameters
        seed = p.random_seed if p.random_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        rng = np.random.RandomState(seed)
        cosmo = FlatLambdaCDM(H0=p.cosmology.H0, Om0=p.cosmology.Om0, Ob0=p.cosmology.Ob0)

        inst = get_instrument_spec(p.model_configuration, p.num_pix)
        band = survey_band_config(p.model_configuration)
        num_pix, delta_pix = inst.num_pix, inst.delta_pix
        x_grid, y_grid = util.make_grid(num_pix=num_pix, delta_pix=delta_pix)

        lens_params = self._effective_lens_params(p)
        theta_e = main_einstein_radius(lens_params, p.z_lens, p.z_source, cosmo)
        sub_type = p.substructure.substructure_type
        lens_light = p.lens_light_enabled

        observed, noiseless, sources, kappas = [], [], [], []
        image_meta: List[SingleImageMetadata] = []

        for idx in range(p.num_images):
            realization = build_lens_model(
                lens_params=lens_params,
                z_lens=p.z_lens,
                z_source=p.z_source,
                cosmo=cosmo,
                substructure_type=sub_type.value,
                cdm_params=p.substructure.cdm_params,
                axion_params=p.substructure.axion_params,
                rng=rng,
                theta_E=theta_e,
            )
            sp = p.source_params
            src_x = sp.center_x if sp.center_x is not None else float(rng.uniform(-sp.position_scatter, sp.position_scatter))
            src_y = sp.center_y if sp.center_y is not None else float(rng.uniform(-sp.position_scatter, sp.position_scatter))

            if band is None:
                rendered = self._render_model_i(p, realization, src_x, src_y, lens_light, num_pix, delta_pix, x_grid, y_grid, rng)
            else:
                rendered = self._render_survey(p, band, realization, src_x, src_y, lens_light, num_pix, x_grid, y_grid, rng)
            image, model, arc, src_plane, exposure, bkg_rms, source_kind = rendered

            snr, peak_snr = signal_to_noise(arc, model, exposure, bkg_rms)
            observed.append(image)
            noiseless.append(model)
            sources.append(src_plane)
            kappas.append(pixelized_kappa(realization, x_grid, y_grid, num_pix, delta_pix))

            notes = [f"Rendered with {inst.name} ({inst.flux_units} units)."]
            if p.model_configuration == ModelConfiguration.MODEL_IV and not sp.galaxy_image_path:
                notes.append("No galaxy image file given; used a procedural spiral galaxy as the source.")
            image_meta.append(SingleImageMetadata(
                image_index=idx,
                model_configuration=p.model_configuration,
                substructure_type=sub_type,
                dimensions=(num_pix, num_pix),
                pixel_scale_arcsec=delta_pix,
                z_lens=p.z_lens,
                z_source=p.z_source,
                einstein_radius_arcsec=round(theta_e, 4),
                min_intensity=round(float(image.min()), 5),
                max_intensity=round(float(image.max()), 5),
                mean_intensity=round(float(image.mean()), 5),
                snr=round(snr, 2),
                peak_snr=round(peak_snr, 2),
                exposure_time_s=round(exposure, 1),
                background_rms=round(bkg_rms, 6),
                source_center_arcsec=(round(src_x, 4), round(src_y, 4)),
                source_type=source_kind,
                lens_light=lens_light,
                subhalo_count=realization.substructure_count,
                substructure_mass_msun=realization.substructure_mass,
                axion_mass_eV=realization.axion_mass_eV,
                notes=notes,
            ))

        elapsed = time.time() - start_time
        summary = (
            f"Generated {p.num_images} image(s) with {p.model_configuration.value} ({inst.name}), "
            f"{sub_type.value} substructure, {num_pix}x{num_pix} px at {delta_pix}\"/px. "
            f"z_lens={p.z_lens}, z_source={p.z_source}, theta_E={theta_e:.3f}\". "
            f"Median arc SNR {np.median([m.snr for m in image_meta]):.1f}. Seed {seed}."
        )
        output = SimulationExecutionOutput(
            plan_id=plan_id,
            success=True,
            num_images_generated=p.num_images,
            model_configuration=p.model_configuration,
            substructure_type=sub_type,
            images_metadata=image_meta,
            random_seed=seed,
            execution_time_seconds=round(elapsed, 3),
            summary=summary,
        )
        return SimulationResultBundle(
            images_observed=np.array(observed),
            images_noiseless=np.array(noiseless),
            source_planes=np.array(sources),
            kappa_maps=np.array(kappas),
            output_metadata=output,
            parameters=p,
        )

    @staticmethod
    def _effective_lens_params(p: SimulationParameters):
        """Model IV uses a fixed theta_E = 1.5" and a round SIE unless the user overrides them."""
        lens = p.lens_params
        if p.model_configuration != ModelConfiguration.MODEL_IV:
            return lens
        update: Dict[str, Any] = {}
        explicit = lens.model_fields_set
        if lens.theta_E is None and lens.velocity_dispersion_kms is None and "mass_solar" not in explicit:
            update["theta_E"] = MODEL_IV_THETA_E
        if "e1" not in explicit:
            update["e1"] = 0.0
        return lens.model_copy(update=update)

    @staticmethod
    def _lens_light_kwargs(p: SimulationParameters, lens_center) -> List[Dict[str, Any]]:
        ll = p.lens_light_params
        cx, cy = lens_center
        return [
            {"magnitude": ll.magnitude, "R_sersic": ll.R_sersic, "n_sersic": ll.n_sersic,
             "e1": 0.0, "e2": 0.05, "center_x": cx, "center_y": cy},
            {"magnitude": ll.halo_magnitude, "R_sersic": 1.5, "n_sersic": 1.2,
             "e1": 0.0, "e2": 0.3, "center_x": cx, "center_y": cy},
        ]

    def _render_model_i(self, p, realization, src_x, src_y, lens_light, num_pix, delta_pix, x_grid, y_grid, rng):
        """DeepLense Model I: amplitude units, Gaussian PSF, exposure ~ 10^U(3, 3.5) s, background RMS 0.01."""
        inst = get_instrument_spec(ModelConfiguration.MODEL_I, num_pix)
        exposure = float(10 ** rng.uniform(*MODEL_I_LOG10_EXPOSURE_RANGE))
        _, _, ra0, dec0, _, _, pix2coord, _ = util.make_grid_with_coordtransform(
            num_pix=num_pix, delta_pix=delta_pix, subgrid_res=1, inverse=False
        )
        data = ImageData(
            background_rms=inst.background_rms, exposure_time=exposure,
            ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=pix2coord,
            image_data=np.zeros((num_pix, num_pix)),
        )
        psf = PSF(psf_type="GAUSSIAN", fwhm=inst.fwhm, pixel_size=delta_pix, truncation=3)

        sp = p.source_params
        kwargs_source = [{"amp": sp.amp, "R_sersic": sp.R_sersic, "n_sersic": sp.n_sersic,
                          "e1": sp.e1, "e2": sp.e2, "center_x": src_x, "center_y": src_y}]
        source_model = LightModel(["SERSIC_ELLIPSE"])
        kwargs_lens_light, lens_light_model = None, None
        if lens_light:
            # Amplitudes scaled relative to the source using the magnitude difference
            kwargs_lens_light = []
            for comp in self._lens_light_kwargs(p, (p.lens_params.center_x, p.lens_params.center_y)):
                comp = dict(comp)
                comp["amp"] = sp.amp * 10 ** (-0.4 * (comp.pop("magnitude") - sp.magnitude))
                kwargs_lens_light.append(comp)
            lens_light_model = LightModel(["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"])

        im = ImageModel(
            data_class=data, psf_class=psf,
            lens_model_class=LensModel(realization.model_list),
            source_model_class=source_model,
            lens_light_model_class=lens_light_model,
            kwargs_numerics=KWARGS_NUMERICS,
        )
        model = im.image(realization.kwargs_list, kwargs_source, kwargs_lens_light)
        arc = im.image(realization.kwargs_list, kwargs_source, kwargs_lens_light, lens_light_add=False) if lens_light else model
        image = add_noise(model, exposure, inst.background_rms, rng)
        src_plane = util.array2image(source_model.surface_brightness(x_grid, y_grid, kwargs_source))
        return image, model, arc, src_plane, exposure, inst.background_rms, "sersic"

    def _render_survey(self, p, band, realization, src_x, src_y, lens_light, num_pix, x_grid, y_grid, rng):
        """Models II-IV: lenstronomy SimAPI with Euclid VIS or HST WFC3 observation settings."""
        sp = p.source_params
        if p.model_configuration == ModelConfiguration.MODEL_IV:
            galaxy = draw_source_galaxy(rng, sp.galaxy_image_path)
            source_list = ["INTERPOL"]
            kwargs_source_mag = [{
                "magnitude": sp.magnitude, "image": galaxy,
                "scale": sp.galaxy_angular_size / galaxy.shape[0],
                "phi_G": float(rng.uniform(-np.pi, np.pi)),
                "center_x": src_x, "center_y": src_y,
            }]
            source_kind = "galaxy_image" if sp.galaxy_image_path else "procedural_galaxy"
        else:
            source_list = ["SERSIC_ELLIPSE"]
            kwargs_source_mag = [{"magnitude": sp.magnitude, "R_sersic": sp.R_sersic, "n_sersic": sp.n_sersic,
                                  "e1": sp.e1, "e2": sp.e2, "center_x": src_x, "center_y": src_y}]
            source_kind = "sersic"

        kwargs_model = {"lens_model_list": realization.model_list, "source_light_model_list": source_list}
        kwargs_lens_light_mag = None
        if lens_light:
            kwargs_model["lens_light_model_list"] = ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"]
            kwargs_lens_light_mag = self._lens_light_kwargs(p, (p.lens_params.center_x, p.lens_params.center_y))

        sim = SimAPI(num_pix, band, kwargs_model)
        kwargs_lens_light, kwargs_source, _ = sim.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag)
        im = sim.image_model_class(KWARGS_NUMERICS)
        model = im.image(realization.kwargs_list, kwargs_source, kwargs_lens_light)
        arc = im.image(realization.kwargs_list, kwargs_source, kwargs_lens_light, lens_light_add=False) if lens_light else model

        exposure = float(sim.exposure_time)
        bkg_rms = float(sim.background_noise)
        image = add_noise(model, exposure, bkg_rms, rng)
        src_plane = util.array2image(sim.source_model_class.surface_brightness(x_grid, y_grid, kwargs_source))
        return image, model, arc, src_plane, exposure, bkg_rms, source_kind
