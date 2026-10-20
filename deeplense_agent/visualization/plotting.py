"""
Visualization and diagnostic plotting module for DeepLense simulations.
Generates multi-panel astrophysical figures and astronomical exports.
"""

from typing import Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from deeplense_agent.simulation.core import SimulationResultBundle


def plot_lensing_diagnostic(
    bundle: SimulationResultBundle,
    image_idx: int = 0,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Renders an astrophysical diagnostic 4-panel figure:
    1. Observed Telescope Image (with PSF & noise)
    2. Dark Matter Convergence Map (kappa) showing substructures
    3. Unlensed Source Galaxy Plane
    4. 1D Radial Surface Brightness Profile
    """
    img = bundle.images_observed[image_idx]
    kappa = bundle.kappa_maps[image_idx]
    src = bundle.source_planes[image_idx]
    meta = bundle.output_metadata.images_metadata[image_idx]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#64748b")

    extent = [
        -meta.dimensions[1] * meta.pixel_scale_arcsec / 2,
        meta.dimensions[1] * meta.pixel_scale_arcsec / 2,
        -meta.dimensions[0] * meta.pixel_scale_arcsec / 2,
        meta.dimensions[0] * meta.pixel_scale_arcsec / 2,
    ]

    # Panel 1: Observed Mock Image
    im0 = axes[0].imshow(img, origin="lower", cmap="inferno", extent=extent)
    axes[0].set_title(
        f"Mock Observation ({meta.model_configuration.value})\n"
        f"Substructure: {meta.substructure_type.value.upper()} | SNR: {meta.estimated_snr:.1f}",
        color="white",
        fontsize=11,
        pad=10,
    )
    axes[0].set_xlabel("Arcseconds", color="white", fontsize=9)
    axes[0].set_ylabel("Arcseconds", color="white", fontsize=9)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb0.ax.axes, "yticklabels"), color="white")

    # Panel 2: Convergence (kappa) Map
    # Log scale or symlog to clearly highlight subtle subhalo perturbations
    v_min = max(0.01, float(np.percentile(kappa, 5)))
    v_max = float(np.percentile(kappa, 99.5))
    im1 = axes[1].imshow(
        kappa,
        origin="lower",
        cmap="magma",
        extent=extent,
        norm=LogNorm(vmin=v_min, vmax=v_max) if v_min < v_max else None,
    )
    axes[1].set_title(
        f"Dark Matter Convergence (κ)\n"
        f"Subhalo count: {meta.subhalo_count} | θ_E: {meta.einstein_radius_arcsec:.3f}\"",
        color="white",
        fontsize=11,
        pad=10,
    )
    axes[1].set_xlabel("Arcseconds", color="white", fontsize=9)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb1.ax.axes, "yticklabels"), color="white")

    # Panel 3: Source Plane Galaxy (unlensed)
    im2 = axes[2].imshow(src, origin="lower", cmap="viridis", extent=extent)
    axes[2].set_title(
        f"Source Plane (Unlensed)\nz_src = {meta.z_source} | z_lens = {meta.z_lens}",
        color="white",
        fontsize=11,
        pad=10,
    )
    axes[2].set_xlabel("Arcseconds", color="white", fontsize=9)
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="white")

    # Panel 4: 1D Radial Profile
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y_idx, x_idx = np.indices((ny, nx))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) * meta.pixel_scale_arcsec
    r_flat = r.ravel()
    flux_flat = img.ravel()
    sort_idx = np.argsort(r_flat)

    # Bin radial profile
    bins = np.linspace(0, np.max(r_flat) * 0.85, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(r_flat, bins)
    bin_means = [flux_flat[digitized == i].mean() for i in range(1, len(bins))]

    axes[3].scatter(r_flat[::4], flux_flat[::4], s=3, color="#38bdf8", alpha=0.3, label="Pixel Flux")
    axes[3].plot(bin_centers, bin_means, color="#f43f5e", lw=2, label="Radial Mean")
    axes[3].axvline(
        meta.einstein_radius_arcsec,
        color="#fbbf24",
        linestyle="--",
        label=f"Einstein Ring ({meta.einstein_radius_arcsec:.2f}\")",
    )
    axes[3].set_title("Surface Brightness Profile", color="white", fontsize=11, pad=10)
    axes[3].set_xlabel("Radius from Center (arcsec)", color="white", fontsize=9)
    axes[3].set_ylabel("Intensity (counts)", color="white", fontsize=9)
    axes[3].legend(loc="upper right", facecolor="#1e293b", edgecolor="#64748b", labelcolor="white")
    axes[3].grid(color="#334155", linestyle=":", alpha=0.6)

    plt.suptitle(
        f"DeepLense Lensing Simulation Diagnostic {title_suffix}",
        color="white",
        fontsize=14,
        y=1.03,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    if not show_plot:
        plt.close(fig)

    return fig


def plot_batch_comparison(
    bundles_dict: dict,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Renders a side-by-side comparison of different DeepLense models or substructures.
    bundles_dict: {label: SimulationResultBundle}
    """
    n_models = len(bundles_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0f172a")

    for ax, (label, b) in zip(axes, bundles_dict.items()):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#64748b")

        img = b.primary_image
        meta = b.output_metadata.images_metadata[0]
        extent = [
            -meta.dimensions[1] * meta.pixel_scale_arcsec / 2,
            meta.dimensions[1] * meta.pixel_scale_arcsec / 2,
            -meta.dimensions[0] * meta.pixel_scale_arcsec / 2,
            meta.dimensions[0] * meta.pixel_scale_arcsec / 2,
        ]
        im = ax.imshow(img, origin="lower", cmap="inferno", extent=extent)
        ax.set_title(f"{label}\n({meta.dimensions[0]}x{meta.dimensions[1]} px)", color="white", fontsize=11)
        ax.set_xlabel("Arcseconds", color="white", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    plt.suptitle("DeepLense Multi-Model Comparison", color="white", fontsize=14, y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())

    if not show_plot:
        plt.close(fig)

    return fig
