"""
Astrophysical Dark Matter Substructure Generators for DeepLenseSim.
Supports:
- No Substructure (Smooth Singular Isothermal Ellipsoid + Shear)
- Cold Dark Matter (CDM) subhalos drawn from power-law Subhalo Mass Function (SHMF)
- Wave/Axion Dark Matter (quantum vortex filaments with de Broglie scale)
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G, c, M_sun

from deeplense_agent.models.parameters import (
    LensMacroParameters,
    CDMSubstructureParameters,
    AxionSubstructureParameters,
    CosmologyConfig,
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


def axion_mass_to_de_broglie_arcsec(axion_mass_eV: float) -> float:
    """
    Converts ultra-light axion mass to de Broglie wavelength scale in arcseconds (at z ~ 0.5).
    Following DeepLenseSim calibration: lambda ~ 0.06 * (2 * m / 1e-22)^(-1) arcsec.
    """
    # Normalized to standard 10^-22 eV scale
    ratio = float(axion_mass_eV) / 1e-22
    if ratio <= 0:
        ratio = 1.0
    de_broglie = 0.06 * (2.0 * ratio) ** (-1.0)
    return float(np.clip(de_broglie, 0.01, 1.5))


def build_lens_model(
    lens_params: LensMacroParameters,
    z_lens: float,
    z_source: float,
    cosmo_config: CosmologyConfig,
    substructure_type: str,
    cdm_params: Optional[CDMSubstructureParameters] = None,
    axion_params: Optional[AxionSubstructureParameters] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[float], int]:
    """
    Constructs the lenstronomy lens model lists (lens_model_list, kwargs_lens_list, lens_redshift_list).
    
    Returns:
        lens_model_list: List of profile model names (e.g. ['SIE', 'SHEAR', 'POINT_MASS', ...])
        kwargs_lens_list: List of parameter dicts for each profile
        lens_redshift_list: List of redshifts for each plane
        substructure_count: Total count of substructure elements added
    """
    if rng is None:
        rng = np.random.RandomState()

    cosmo = FlatLambdaCDM(H0=cosmo_config.H0, Om0=cosmo_config.Om0, Ob0=cosmo_config.Ob0)

    # Calculate main halo Einstein radius if not explicitly provided
    if lens_params.theta_E is not None:
        theta_e_main = lens_params.theta_E
    else:
        theta_e_main = compute_einstein_radius(lens_params.mass_solar, z_lens, z_source, cosmo)

    # Base Macro Model: SIE (Singular Isothermal Ellipsoid) + External Shear
    lens_model_list = ["SIE", "SHEAR"]
    kwargs_lens_list = [
        {
            "theta_E": theta_e_main,
            "e1": lens_params.e1,
            "e2": lens_params.e2,
            "center_x": lens_params.center_x,
            "center_y": lens_params.center_y,
        },
        {
            "gamma1": lens_params.gamma1,
            "gamma2": lens_params.gamma2,
        },
    ]
    lens_redshift_list = [z_lens, z_lens]
    substructure_count = 0

    # 1. No Substructure
    if substructure_type in ("no_sub", "none"):
        return lens_model_list, kwargs_lens_list, lens_redshift_list, substructure_count

    # 2. Cold Dark Matter (CDM) Subhalos
    elif substructure_type == "cdm":
        if cdm_params is None:
            cdm_params = CDMSubstructureParameters()

        # Sample number of subhalos from Poisson distribution
        n_halos = int(rng.poisson(cdm_params.n_sub))
        if n_halos > 0:
            # Draw subhalo masses from power-law dN/dm ~ m^beta
            u_vals = rng.uniform(0.0, 1.0, size=n_halos)
            b = cdm_params.beta + 1.0
            m_low_u = cdm_params.m_sub_min ** b
            m_high_u = cdm_params.m_sub_max ** b
            sub_masses = (m_low_u + (m_high_u - m_low_u) * u_vals) ** (1.0 / b)

            # Convert subhalo masses to Einstein radii
            theta_sub = [
                compute_einstein_radius(m, z_lens, z_source, cosmo)
                for m in sub_masses
            ]

            # Distribute subhalos in an annulus around the Einstein ring
            r_annulus = rng.uniform(0.25 * theta_e_main, 1.75 * theta_e_main, size=n_halos)
            phi_annulus = rng.uniform(0.0, 2.0 * np.pi, size=n_halos)
            sub_x = r_annulus * np.cos(phi_annulus) + lens_params.center_x
            sub_y = r_annulus * np.sin(phi_annulus) + lens_params.center_y

            for i in range(n_halos):
                lens_model_list.append("POINT_MASS")
                kwargs_lens_list.append({
                    "theta_E": float(theta_sub[i]),
                    "center_x": float(sub_x[i]),
                    "center_y": float(sub_y[i]),
                })
                lens_redshift_list.append(z_lens)

            substructure_count = n_halos

    # 3. Wave / Axion Dark Matter Vortex
    elif substructure_type == "axion":
        if axion_params is None:
            axion_params = AxionSubstructureParameters()

        res = axion_params.resolution_elements
        sub_element_mass = axion_params.vortex_mass / float(res)
        e_sub = compute_einstein_radius(sub_element_mass, z_lens, z_source, cosmo)

        de_broglie = axion_mass_to_de_broglie_arcsec(axion_params.axion_mass_eV)
        coords = np.linspace(-de_broglie, de_broglie, res)
        ang = rng.uniform(0.0, 2.0 * np.pi)

        # Distribute elements along rotated filament crossing the critical curve
        for x_coord in coords:
            rx = x_coord * np.cos(ang) + lens_params.center_x
            ry = x_coord * np.sin(ang) + lens_params.center_y

            lens_model_list.append("POINT_MASS")
            kwargs_lens_list.append({
                "theta_E": float(e_sub),
                "center_x": float(rx),
                "center_y": float(ry),
            })
            lens_redshift_list.append(z_lens)

        substructure_count = res

    return lens_model_list, kwargs_lens_list, lens_redshift_list, substructure_count
