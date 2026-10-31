"""
Astrophysical Dark Matter Substructure Generators for DeepLenseSim.
Supports:
- No Substructure (Smooth Singular Isothermal Ellipsoid + Shear)
- Cold Dark Matter (CDM) subhalos drawn from power-law Subhalo Mass Function (SHMF)
- Wave/Axion Dark Matter (quantum vortex filaments with de Broglie scale)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G, c, M_sun

from deeplense_agent.models.parameters import (
    LensMacroParameters,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
)


def compute_einstein_radius(
    mass_solar: float,
    z_lens: float,
    z_source: float,
    cosmo: FlatLambdaCDM,
) -> float:
    """
    Computes the Einstein radius in arcseconds for a point mass or SIE lens.
    Uses proper angular diameter distances (D_l, D_s, D_ls).
    
    Formula:
        theta_E = sqrt((4 * G * M / c^2) * (D_ls / (D_l * D_s))) [rad]
    """
    if z_source <= z_lens:
        raise ValueError(f"Source redshift z_s ({z_source}) must exceed lens redshift z_l ({z_lens})")

    # Angular diameter distances in meters
    d_l = cosmo.angular_diameter_distance(z_lens).to(u.m).value
    d_s = cosmo.angular_diameter_distance(z_source).to(u.m).value
    d_ls = cosmo.angular_diameter_distance(z_lens, z_source).to(u.m).value

    m_kg = mass_solar * M_sun.value
    g_val = G.value
    c_val = c.value

    # Physical theta_E in radians
    theta_rad = np.sqrt(4.0 * g_val * m_kg / (c_val ** 2) * (d_ls / (d_l * d_s)))
    rad_to_arcsec = 206264.806247  # arcseconds per radian

    return float(theta_rad * rad_to_arcsec)


def isothermal_einstein_radius(
    velocity_dispersion_kms: float,
    z_lens: float,
    z_source: float,
    cosmo: FlatLambdaCDM,
) -> float:
    """
    Einstein radius in arcsec of a singular isothermal sphere:
        theta_E = 4 pi (sigma_v / c)^2 * D_ls / D_s
    """
    if z_source <= z_lens:
        raise ValueError(f"Source redshift z_s ({z_source}) must exceed lens redshift z_l ({z_lens})")
    d_s = cosmo.angular_diameter_distance(z_source).value
    d_ls = cosmo.angular_diameter_distance(z_lens, z_source).value
    theta_rad = 4.0 * np.pi * (velocity_dispersion_kms * 1e3 / c.value) ** 2 * d_ls / d_s
    return float(theta_rad * 206264.806247)


def main_einstein_radius(
    lens_params: LensMacroParameters,
    z_lens: float,
    z_source: float,
    cosmo: FlatLambdaCDM,
) -> float:
    """Main deflector theta_E: explicit value, else velocity dispersion, else halo mass."""
    if lens_params.theta_E is not None:
        return float(lens_params.theta_E)
    if lens_params.velocity_dispersion_kms is not None:
        return isothermal_einstein_radius(lens_params.velocity_dispersion_kms, z_lens, z_source, cosmo)
    return compute_einstein_radius(lens_params.mass_solar, z_lens, z_source, cosmo)


def axion_mass_to_de_broglie_arcsec(axion_mass_eV: float) -> float:
    """
    Vortex half-length in arcsec for an ultra-light axion mass, using the DeepLenseSim
    calibration (valid for a halo at z ~ 0.5): lambda = 0.06 * (2 m / 1e-22 eV)^-1.
    """
    ratio = float(axion_mass_eV) / 1e-22
    if ratio <= 0:
        ratio = 1.0
    de_broglie = 0.06 * (2.0 * ratio) ** (-1.0)
    return float(np.clip(de_broglie, 0.01, 1.5))


def sample_axion_mass(rng: np.random.RandomState) -> float:
    """Axion mass prior used by DeepLenseSim: log-uniform between 1e-24 and 1e-22 eV."""
    return float(10 ** rng.uniform(-24.0, -22.0))


class LensModelRealization:
    """One random realization of the lens: lenstronomy model lists plus bookkeeping for metadata."""
    def __init__(self, model_list, kwargs_list, theta_E, substructure_count=0, substructure_mass=0.0, axion_mass_eV=None):
        self.model_list: List[str] = model_list
        self.kwargs_list: List[Dict[str, Any]] = kwargs_list
        self.theta_E: float = theta_E
        self.substructure_count: int = substructure_count
        self.substructure_mass: float = substructure_mass
        self.axion_mass_eV: Optional[float] = axion_mass_eV


def build_lens_model(
    lens_params: LensMacroParameters,
    z_lens: float,
    z_source: float,
    cosmo: FlatLambdaCDM,
    substructure_type: str,
    cdm_params: Optional[CDMSubstructureParameters] = None,
    axion_params: Optional[AxionSubstructureParameters] = None,
    rng: Optional[np.random.RandomState] = None,
    theta_E: Optional[float] = None,
) -> LensModelRealization:
    """
    Builds an SIE + external shear macro model and adds the requested substructure.

    CDM: point-mass subhalos with masses from dN/dm ~ m^beta, placed at radius
    r ~ U(r_min, r_max) arcsec and uniform angle (DeepLenseSim `make_old_cdm`).
    Axion: a straight vortex string of point masses through the lens center, with
    half-length set by the de Broglie scale (DeepLenseSim `make_vortex`).
    """
    if rng is None:
        rng = np.random.RandomState()
    theta_e_main = theta_E if theta_E is not None else main_einstein_radius(lens_params, z_lens, z_source, cosmo)

    model_list = ["SIE", "SHEAR"]
    kwargs_list: List[Dict[str, Any]] = [
        {
            "theta_E": theta_e_main,
            "e1": lens_params.e1,
            "e2": lens_params.e2,
            "center_x": lens_params.center_x,
            "center_y": lens_params.center_y,
        },
        {"gamma1": lens_params.gamma1, "gamma2": lens_params.gamma2},
    ]
    realization = LensModelRealization(model_list, kwargs_list, theta_e_main)
    # Point-mass theta_E scales as sqrt(M): compute the distance factor once
    theta_per_sqrt_msun = compute_einstein_radius(1.0, z_lens, z_source, cosmo)

    if substructure_type == "cdm":
        cdm = cdm_params or CDMSubstructureParameters()
        n_halos = int(rng.poisson(cdm.n_sub)) if cdm.poisson_count else cdm.n_sub
        if n_halos > 0:
            # Inverse-CDF sampling of a truncated power law (log-uniform when beta = -1)
            b = cdm.beta + 1.0
            u_vals = rng.uniform(0.0, 1.0, size=n_halos)
            if abs(b) < 1e-9:
                masses = cdm.m_sub_min * (cdm.m_sub_max / cdm.m_sub_min) ** u_vals
            else:
                lo, hi = cdm.m_sub_min ** b, cdm.m_sub_max ** b
                masses = (lo + (hi - lo) * u_vals) ** (1.0 / b)
            r = rng.uniform(cdm.r_min, cdm.r_max, size=n_halos)
            phi = rng.uniform(0.0, 2.0 * np.pi, size=n_halos)
            for m, ri, ph in zip(masses, r, phi):
                model_list.append("POINT_MASS")
                kwargs_list.append({
                    "theta_E": float(theta_per_sqrt_msun * np.sqrt(m)),
                    "center_x": float(ri * np.cos(ph) + lens_params.center_x),
                    "center_y": float(ri * np.sin(ph) + lens_params.center_y),
                })
            realization.substructure_count = n_halos
            realization.substructure_mass = float(np.sum(masses))

    elif substructure_type == "axion":
        axion = axion_params or AxionSubstructureParameters()
        m_a = axion.axion_mass_eV if axion.axion_mass_eV is not None else sample_axion_mass(rng)
        res = axion.resolution_elements
        theta_element = float(theta_per_sqrt_msun * np.sqrt(axion.vortex_mass / res))
        half_length = axion_mass_to_de_broglie_arcsec(m_a)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        for t in np.linspace(-half_length, half_length, res):
            model_list.append("POINT_MASS")
            kwargs_list.append({
                "theta_E": theta_element,
                "center_x": float(t * np.cos(ang) + lens_params.center_x),
                "center_y": float(t * np.sin(ang) + lens_params.center_y),
            })
        realization.substructure_count = res
        realization.substructure_mass = float(axion.vortex_mass)
        realization.axion_mass_eV = m_a

    return realization
