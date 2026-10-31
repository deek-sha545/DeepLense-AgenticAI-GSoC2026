"""
Pydantic Models for DeepLense Gravitational Lensing Simulation Parameters.
Provides strict physical validation and serialization for strong lensing configurations.

Defaults follow the reference DeepLenseSim pipeline (github.com/mwt5345/DeepLenseSim):
z_lens = 0.5, z_source = 1.0, SIE main halo of 1e12 M_sun with e1 = 0.1 and 0.05 shear,
25 point-mass CDM subhalos and 100-element axion vortex strings.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class SubstructureType(str, Enum):
    """Dark matter substructure models supported in DeepLense."""
    NO_SUB = "no_sub"
    CDM = "cdm"
    AXION = "axion"


class ModelConfiguration(str, Enum):
    """
    Simulation configurations matching the DeepLense datasets:
    - MODEL_I: Generic optical benchmark (150x150 pix, 0.05"/pix, Gaussian PSF, SNR ~ 25)
    - MODEL_II: Euclid VIS (64x64 pix, lenstronomy Euclid observation config)
    - MODEL_III: HST WFC3 (64x64 pix, lenstronomy HST observation config)
    - MODEL_IV: Euclid VIS with a real-galaxy (interpolated image) source and lens galaxy light
    """
    MODEL_I = "Model_I"
    MODEL_II = "Model_II"
    MODEL_III = "Model_III"
    MODEL_IV = "Model_IV"


class InstrumentSpecification(BaseModel):
    """Resolved instrument and detector description for one model configuration."""
    name: str = Field(..., description="Instrument or survey name")
    num_pix: int = Field(..., ge=32, le=512, description="Image dimension (num_pix x num_pix)")
    delta_pix: float = Field(..., gt=0.001, le=0.5, description="Pixel scale in arcsec/pixel")
    fwhm: float = Field(..., gt=0.001, description="Gaussian PSF FWHM in arcsec")
    exposure_time: float = Field(..., gt=0.0, description="Total exposure time in seconds")
    background_rms: Optional[float] = Field(
        default=None, gt=0.0,
        description="Background RMS per pixel in counts/s (Model I); survey models derive it from sky and read noise",
    )
    flux_units: str = Field(default="magnitude", description="'amplitude' (Model I) or 'magnitude' (survey models)")


class LensMacroParameters(BaseModel):
    """
    Primary deflector (Singular Isothermal Ellipsoid + external shear).

    The Einstein radius is taken from, in order of precedence: `theta_E`,
    `velocity_dispersion_kms` (isothermal relation), or `mass_solar` (point-mass
    relation, as used by DeepLenseSim).
    """
    mass_solar: float = Field(default=1e12, gt=1e9, le=1e15, description="Main halo mass in M_sun used for theta_E")
    velocity_dispersion_kms: Optional[float] = Field(
        default=None, ge=50.0, le=500.0, description="Velocity dispersion in km/s; sets theta_E = 4 pi (sigma/c)^2 D_ls/D_s"
    )
    theta_E: Optional[float] = Field(default=None, gt=0.1, le=10.0, description="Einstein radius in arcsec (overrides mass and sigma)")
    e1: float = Field(default=0.1, ge=-0.8, le=0.8, description="Ellipticity component 1")
    e2: float = Field(default=0.0, ge=-0.8, le=0.8, description="Ellipticity component 2")
    center_x: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center X in arcsec")
    center_y: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center Y in arcsec")
    gamma1: float = Field(default=0.05, ge=-0.5, le=0.5, description="External shear component 1")
    gamma2: float = Field(default=0.0, ge=-0.5, le=0.5, description="External shear component 2")


class SourceLightParameters(BaseModel):
    """
    Background source galaxy light.

    Models I-III use an elliptical Sersic profile. Model IV interpolates a galaxy
    image: a Galaxy10 DECaLS .h5 file, a single .npy/.fits/.png image, or a
    procedurally generated spiral if `galaxy_image_path` is not set.
    """
    magnitude: float = Field(default=20.0, ge=12.0, le=30.0, description="Apparent magnitude (survey models II-IV)")
    amp: float = Field(default=20.0, gt=0.0, description="Sersic amplitude in counts/s (Model I)")
    R_sersic: float = Field(default=0.25, gt=0.01, le=5.0, description="Sersic half-light radius in arcsec")
    n_sersic: float = Field(default=1.0, ge=0.5, le=8.0, description="Sersic index (1 = exponential disk, 4 = de Vaucouleurs)")
    e1: float = Field(default=-0.1, ge=-0.8, le=0.8, description="Source ellipticity component 1")
    e2: float = Field(default=0.1, ge=-0.8, le=0.8, description="Source ellipticity component 2")
    center_x: Optional[float] = Field(default=None, ge=-1.5, le=1.5, description="Source X offset in arcsec; random per image if unset")
    center_y: Optional[float] = Field(default=None, ge=-1.5, le=1.5, description="Source Y offset in arcsec; random per image if unset")
    position_scatter: float = Field(default=0.35, ge=0.0, le=1.5, description="Half-width of the uniform prior on random source offsets (arcsec)")
    galaxy_image_path: Optional[str] = Field(default=None, description="Model IV source: Galaxy10_DECals.h5, or a .npy/.fits/.png image")
    galaxy_angular_size: float = Field(default=1.0, gt=0.05, le=5.0, description="Model IV source: angular size of the galaxy image in arcsec")


class LensLightParameters(BaseModel):
    """Deflector galaxy light (two elliptical Sersic components, as in DeepLense Model IV)."""
    magnitude: float = Field(default=18.0, ge=12.0, le=30.0, description="Bulge apparent magnitude")
    R_sersic: float = Field(default=0.4, gt=0.01, le=5.0, description="Bulge half-light radius in arcsec")
    n_sersic: float = Field(default=2.3, ge=0.5, le=8.0, description="Bulge Sersic index")
    halo_magnitude: float = Field(default=28.0, ge=12.0, le=35.0, description="Faint extended component magnitude")


class CDMSubstructureParameters(BaseModel):
    """Cold Dark Matter subhalos as point masses drawn from a power-law mass function."""
    n_sub: int = Field(default=25, ge=1, le=200, description="Number of subhalos (exact, or Poisson mean if poisson_count)")
    poisson_count: bool = Field(default=False, description="Draw the subhalo count from a Poisson distribution")
    m_sub_min: float = Field(default=1e6, gt=1e3, le=1e9, description="Minimum subhalo mass in M_sun")
    m_sub_max: float = Field(default=1e10, gt=1e7, le=1e12, description="Maximum subhalo mass in M_sun")
    beta: float = Field(default=-0.9, ge=-2.5, le=-0.1, description="Mass function slope, dN/dm ~ m^beta (DeepLenseSim uses -0.9; CDM theory ~ -1.9)")
    r_min: float = Field(default=0.25, ge=0.0, le=5.0, description="Inner radius of subhalo placement in arcsec")
    r_max: float = Field(default=2.0, gt=0.0, le=5.0, description="Outer radius of subhalo placement in arcsec")

    @model_validator(mode="after")
    def validate_ranges(self) -> "CDMSubstructureParameters":
        if self.m_sub_min >= self.m_sub_max:
            raise ValueError(f"m_sub_min ({self.m_sub_min:.1e}) must be strictly less than m_sub_max ({self.m_sub_max:.1e})")
        if self.r_min >= self.r_max:
            raise ValueError(f"r_min ({self.r_min}) must be less than r_max ({self.r_max})")
        return self


class AxionSubstructureParameters(BaseModel):
    """Ultra-light axion dark matter: a vortex string of point masses through the lens center."""
    axion_mass_eV: Optional[float] = Field(
        default=None, ge=1e-25, le=1e-19,
        description="Axion mass in eV; drawn log-uniformly from 1e-24 to 1e-22 per image if unset",
    )
    vortex_mass: float = Field(default=3e10, gt=1e7, le=1e12, description="Total vortex mass in M_sun")
    resolution_elements: int = Field(default=100, ge=10, le=500, description="Point masses used to discretize the vortex")


class SubstructureConfig(BaseModel):
    """Composite substructure configuration."""
    substructure_type: SubstructureType = Field(default=SubstructureType.NO_SUB, description="Type of substructure")
    cdm_params: Optional[CDMSubstructureParameters] = Field(default=None, description="Parameters when type is CDM")
    axion_params: Optional[AxionSubstructureParameters] = Field(default=None, description="Parameters when type is AXION")

    @model_validator(mode="after")
    def auto_populate_substructure_params(self) -> "SubstructureConfig":
        if self.substructure_type == SubstructureType.CDM and self.cdm_params is None:
            self.cdm_params = CDMSubstructureParameters()
        elif self.substructure_type == SubstructureType.AXION and self.axion_params is None:
            self.axion_params = AxionSubstructureParameters()
        return self


class CosmologyConfig(BaseModel):
    """Flat Lambda-CDM cosmology for distance calculations."""
    H0: float = Field(default=70.0, ge=50.0, le=100.0, description="Hubble constant in km/s/Mpc")
    Om0: float = Field(default=0.3, ge=0.05, le=1.0, description="Total matter density parameter Omega_m")
    Ob0: float = Field(default=0.05, ge=0.01, le=0.2, description="Baryon density parameter Omega_b")


class SimulationParameters(BaseModel):
    """
    Complete configuration for a DeepLense strong lensing simulation.
    Includes physics validation guardrails for redshifts and geometry.
    """
    model_configuration: ModelConfiguration = Field(
        default=ModelConfiguration.MODEL_II,
        description="Model configuration (Model_I, Model_II, Model_III, Model_IV)"
    )
    substructure: SubstructureConfig = Field(
        default_factory=SubstructureConfig,
        description="Substructure configuration (no_sub, cdm, axion)"
    )
    z_lens: float = Field(default=0.5, ge=0.01, le=5.0, description="Deflector / lens redshift")
    z_source: float = Field(default=1.0, ge=0.05, le=10.0, description="Background source galaxy redshift")
    num_images: int = Field(default=1, ge=1, le=50, description="Number of lensing images to generate in batch")
    lens_params: LensMacroParameters = Field(default_factory=LensMacroParameters, description="Macro lens parameters")
    source_params: SourceLightParameters = Field(default_factory=SourceLightParameters, description="Source light parameters")
    include_lens_light: Optional[bool] = Field(
        default=None, description="Add deflector galaxy light; defaults to True for Model IV only, as in DeepLense"
    )
    lens_light_params: LensLightParameters = Field(default_factory=LensLightParameters, description="Deflector light parameters")
    cosmology: CosmologyConfig = Field(default_factory=CosmologyConfig, description="Cosmology model")
    num_pix: Optional[int] = Field(default=None, ge=32, le=512, description="Override the model's image size in pixels")
    random_seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1, description="Random seed; one is drawn and recorded if unset")

    @model_validator(mode="after")
    def validate_astrophysics(self) -> "SimulationParameters":
        if self.z_lens >= self.z_source:
            raise ValueError(
                f"Astrophysical violation: Lens redshift z_lens ({self.z_lens}) "
                f"must be strictly less than source redshift z_source ({self.z_source})."
            )
        if (self.z_source - self.z_lens) < 0.05:
            raise ValueError(
                f"Redshift difference (z_source - z_lens = {self.z_source - self.z_lens:.3f}) is too small "
                "for significant gravitational lensing deflection."
            )
        return self

    @property
    def lens_light_enabled(self) -> bool:
        if self.include_lens_light is not None:
            return self.include_lens_light
        return self.model_configuration == ModelConfiguration.MODEL_IV
