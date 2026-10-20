"""
Pydantic Models for DeepLense Gravitational Lensing Simulation Parameters.
Provides strict physical validation and serialization for strong lensing configurations.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator, field_validator


class SubstructureType(str, Enum):
    """Dark matter substructure models supported in DeepLense."""
    NO_SUB = "no_sub"
    CDM = "cdm"
    AXION = "axion"


class ModelConfiguration(str, Enum):
    """
    Simulation configurations matching DeepLenseSim papers:
    - MODEL_I: Ground-based / standard benchmark (150x150 pix, Gaussian PSF)
    - MODEL_II: Space-based Euclid VIS survey approximation (64x64 pix)
    - MODEL_III: Space-based HST survey approximation (64x64 pix)
    - MODEL_IV: Space-based Euclid with realistic / complex galaxy morphology
    """
    MODEL_I = "Model_I"
    MODEL_II = "Model_II"
    MODEL_III = "Model_III"
    MODEL_IV = "Model_IV"


class InstrumentSpecification(BaseModel):
    """Instrument and detector specifications for lensing observations."""
    name: str = Field(default="Custom", description="Name of instrument or survey (e.g. Ground, Euclid, HST)")
    num_pix: int = Field(default=64, ge=32, le=512, description="Image dimension (num_pix x num_pix)")
    delta_pix: float = Field(default=0.05, gt=0.001, le=0.5, description="Pixel scale in arcsec/pixel")
    psf_type: str = Field(default="GAUSSIAN", description="PSF model type (GAUSSIAN, PIXEL)")
    fwhm: float = Field(default=0.087, gt=0.001, description="Full-Width at Half-Maximum of PSF in arcsec")
    exposure_time: float = Field(default=2000.0, gt=0.0, description="Exposure time in seconds")
    background_rms: float = Field(default=0.01, gt=0.0, description="Background noise RMS per pixel")


class LensMacroParameters(BaseModel):
    """Primary deflector galaxy (Singular Isothermal Ellipsoid + Shear)."""
    mass_solar: float = Field(default=1e12, gt=1e9, le=1e15, description="Main halo total mass in solar masses M_sun")
    theta_E: Optional[float] = Field(default=None, gt=0.1, le=10.0, description="Einstein radius in arcsec (auto-calculated from mass and redshifts if None)")
    e1: float = Field(default=0.1, ge=-0.8, le=0.8, description="Ellipticity component 1")
    e2: float = Field(default=0.0, ge=-0.8, le=0.8, description="Ellipticity component 2")
    center_x: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center X in arcsec")
    center_y: float = Field(default=0.0, ge=-2.0, le=2.0, description="Lens center Y in arcsec")
    gamma1: float = Field(default=0.05, ge=-0.5, le=0.5, description="External shear component 1")
    gamma2: float = Field(default=0.0, ge=-0.5, le=0.5, description="External shear component 2")


class SourceLightParameters(BaseModel):
    """Background source galaxy light profile (Sérsic profile)."""
    magnitude: Optional[float] = Field(default=20.0, ge=12.0, le=30.0, description="Source apparent magnitude (Model II/III)")
    amp: Optional[float] = Field(default=20.0, gt=0.0, description="Source amplitude intensity (Model I)")
    R_sersic: float = Field(default=0.25, gt=0.01, le=5.0, description="Sérsic half-light radius in arcsec")
    n_sersic: float = Field(default=1.0, ge=0.5, le=8.0, description="Sérsic index (1.0 = exponential disk, 4.0 = de Vaucouleurs)")
    e1: float = Field(default=-0.1, ge=-0.8, le=0.8, description="Source ellipticity component 1")
    e2: float = Field(default=0.1, ge=-0.8, le=0.8, description="Source ellipticity component 2")
    center_x: float = Field(default=0.05, ge=-1.5, le=1.5, description="Source position X offset in arcsec")
    center_y: float = Field(default=0.05, ge=-1.5, le=1.5, description="Source position Y offset in arcsec")


class CDMSubstructureParameters(BaseModel):
    """Cold Dark Matter (CDM) subhalo population parameters."""
    n_sub: int = Field(default=25, ge=1, le=200, description="Mean number of subhalos in field of view")
    m_sub_min: float = Field(default=1e6, gt=1e3, le=1e9, description="Minimum subhalo mass in M_sun")
    m_sub_max: float = Field(default=1e10, gt=1e7, le=1e12, description="Maximum subhalo mass in M_sun")
    beta: float = Field(default=-0.9, ge=-2.5, le=-0.1, description="Power-law slope of subhalo mass function (SHMF)")

    @model_validator(mode="after")
    def validate_mass_range(self) -> "CDMSubstructureParameters":
        if self.m_sub_min >= self.m_sub_max:
            raise ValueError(f"m_sub_min ({self.m_sub_min:.1e}) must be strictly less than m_sub_max ({self.m_sub_max:.1e})")
        return self


class AxionSubstructureParameters(BaseModel):
    """Ultra-light axion / Wave Dark Matter (vortex substructure) parameters."""
    axion_mass_eV: float = Field(default=1e-22, ge=1e-25, le=1e-19, description="Axion particle mass in eV")
    vortex_mass: float = Field(default=3e10, gt=1e7, le=1e12, description="Total vortex mass perturbation in M_sun")
    resolution_elements: int = Field(default=100, ge=10, le=500, description="Discretization elements for the vortex filament")


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
    """Cosmological parameters for distance scale calculations."""
    H0: float = Field(default=70.0, ge=50.0, le=100.0, description="Hubble constant in km/s/Mpc")
    Om0: float = Field(default=0.3, ge=0.05, le=1.0, description="Total matter density parameter Omega_m")
    Ob0: float = Field(default=0.05, ge=0.01, le=0.2, description="Baryon density parameter Omega_b")


class SimulationParameters(BaseModel):
    """
    Complete configuration for DeepLense strong gravitational lensing simulation.
    Includes physics validation guardrails for redshifts, geometry, and instrument configs.
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
    z_source: float = Field(default=1.5, ge=0.05, le=10.0, description="Background source galaxy redshift")
    num_images: int = Field(default=1, ge=1, le=50, description="Number of lensing images to generate in batch")
    lens_params: LensMacroParameters = Field(default_factory=LensMacroParameters, description="Macro lens parameters")
    source_params: SourceLightParameters = Field(default_factory=SourceLightParameters, description="Source light parameters")
    cosmology: CosmologyConfig = Field(default_factory=CosmologyConfig, description="Cosmology model")
    instrument_override: Optional[InstrumentSpecification] = Field(
        default=None,
        description="Optional custom instrument settings overriding default Model settings"
    )
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    @model_validator(mode="after")
    def validate_astrophysics(self) -> "SimulationParameters":
        # Check cosmological redshift ordering
        if self.z_lens >= self.z_source:
            raise ValueError(
                f"Astrophysical violation: Lens redshift z_lens ({self.z_lens}) "
                f"must be strictly less than source redshift z_source ({self.z_source})."
            )
        
        # Enforce minimum separation for gravitational lensing geometry
        if (self.z_source - self.z_lens) < 0.05:
            raise ValueError(
                f"Redshift difference (z_source - z_lens = {self.z_source - self.z_lens:.3f}) is too small "
                "for significant gravitational lensing deflection."
            )
        return self
