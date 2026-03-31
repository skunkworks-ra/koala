"""
apclean.diagnostics
===================
Multi-page diagnostic PDF.  Called from APClean.run(debug=True).

Pages
-----
1  Convergence    — log-peak vs cumulative iteration, major-cycle boundaries,
                   CLEAN iterations bar chart
2  I residual MFS — dirty + after each major cycle (side-by-side or grid)
3  Q / U residuals — dirty MFS vs final MFS for each Stokes
4  Final products  — restored I, Q, U, P, fractional pol  (2×3 grid)
5  VROOM spectra   — dirty Q/I, U/I vs VROOM model at source pixel
6  Minor traces    — per-major-cycle peak vs iteration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# ── shared colour maps ──────────────────────────────────────────────────────
_CMAP_I  = "inferno"
_CMAP_QU = "RdBu_r"
_CMAP_P  = "plasma"


# ============================================================================
# Entry point
# ============================================================================

def plot_all(
    diag        : dict,
    wcs_2d      ,
    freq_hz     : np.ndarray,
    prefix      : str,
) -> None:
    """
    Write a multi-page diagnostic PDF to ``{prefix}_diagnostic.pdf``.
    """
    pdf_path = f"{prefix}_diagnostic.pdf"
    logger.info(f"Writing diagnostic PDF → {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        _page_convergence(pdf, diag)
        _page_i_residuals(pdf, diag)
        _page_qu_residuals(pdf, diag)
        _page_final_products(pdf, diag, wcs_2d, freq_hz)
        _page_vroom_spectra(pdf, diag, freq_hz)
        _page_minor_traces(pdf, diag)

    logger.info(f"Diagnostic PDF written: {pdf_path}")


# ============================================================================
# Page helpers
# ============================================================================

def _page_convergence(pdf: PdfPages, diag: dict) -> None:
    cycles = diag.get("major_cycles", [])
    if not cycles:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("APCLEAN — Convergence", fontsize=13, fontweight="bold")

    # --- left: log peak vs cumulative iterations -------------------------
    ax = axes[0]
    cum_iter = 0
    boundaries = [0]
    peaks_before = []
    peaks_after  = []
    x_vals_b = []
    x_vals_a = []

    for cyc in cycles:
        peaks_before.append(cyc["peak_before"])
        x_vals_b.append(cum_iter)
        cum_iter += cyc["n_clean_iters"]
        peaks_after.append(cyc["peak_after"])
        x_vals_a.append(cum_iter)
        boundaries.append(cum_iter)

    ax.semilogy(x_vals_b, peaks_before, "o", color="royalblue",
                label="Peak before minor", zorder=3)
    ax.semilogy(x_vals_a, peaks_after,  "s", color="tomato",
                label="Peak after minor",  zorder=3)

    # Join with lines
    xs = []
    ys = []
    for b, pb, a, pa in zip(x_vals_b, peaks_before, x_vals_a, peaks_after):
        xs += [b, a]
        ys += [pb, pa]
    ax.semilogy(xs, ys, "-", color="grey", linewidth=0.8, zorder=1)

    # Thresholds
    if cycles:
        ct = cycles[-1]["cycle_thresh"]
        gt = cycles[-1]["global_thresh"]
        ax.axhline(ct, color="orange",  linestyle="--", linewidth=1,
                   label=f"cycle_thresh={ct:.3f}")
        ax.axhline(gt, color="crimson", linestyle=":",  linewidth=1,
                   label=f"global_thresh={gt:.3f}")

    # Major-cycle boundary lines
    for xb in boundaries[1:-1]:
        ax.axvline(xb, color="grey", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Cumulative CLEAN iterations")
    ax.set_ylabel("Residual peak (Jy/beam)")
    ax.set_title("Peak evolution")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.3)

    # --- right: CLEAN iterations bar chart --------------------------------
    ax2 = axes[1]
    cycle_nums = np.arange(1, len(cycles) + 1)
    n_iters    = [c["n_clean_iters"] for c in cycles]
    ax2.bar(cycle_nums, n_iters, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Major cycle")
    ax2.set_ylabel("CLEAN iterations")
    ax2.set_title("Iterations per major cycle")
    ax2.set_xticks(cycle_nums)
    ax2.grid(axis="y", linewidth=0.4)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_i_residuals(pdf: PdfPages, diag: dict) -> None:
    snapshots = diag.get("mfs_snapshots", [])
    if not snapshots:
        return

    n = len(snapshots)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("I MFS Residual Evolution", fontsize=13, fontweight="bold")

    imgs = [s["I"] for s in snapshots]
    vmax = max(np.nanmax(np.abs(img)) for img in imgs)
    vmin = -vmax * 0.1

    labels = ([f"After major {i}" for i in range(1, n)] + ["Final"])[-n:]
    labels[0] = "Dirty"
    if len(labels) < n:
        labels = ["Dirty"] + [f"After major {i}" for i in range(1, n - 1)] + ["Final"]

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        ax = axes[i]
        im = ax.imshow(img, origin="lower", cmap=_CMAP_I,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
        _add_colorbar(ax, im)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_qu_residuals(pdf: PdfPages, diag: dict) -> None:
    snapshots = diag.get("mfs_snapshots", [])
    if not snapshots or len(snapshots) < 2:
        return

    dirty = snapshots[0]
    final = snapshots[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Q / U MFS Residuals — Dirty vs Final", fontsize=13,
                 fontweight="bold")

    stokes_pairs = [("Q", dirty["Q"], final["Q"]),
                    ("U", dirty["U"], final["U"])]

    for row, (label, dirty_img, final_img) in enumerate(stokes_pairs):
        vmax = max(np.nanmax(np.abs(dirty_img)), np.nanmax(np.abs(final_img)))
        for col, (img, title) in enumerate([(dirty_img, f"Dirty {label}"),
                                             (final_img, f"Final {label}")]):
            ax = axes[row, col]
            im = ax.imshow(img, origin="lower", cmap=_CMAP_QU,
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            _add_colorbar(ax, im)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_final_products(
    pdf    : PdfPages,
    diag   : dict,
    wcs_2d ,
    freq_hz: np.ndarray,
) -> None:
    snapshots = diag.get("mfs_snapshots", [])
    if not snapshots:
        return

    final  = snapshots[-1]
    i_mfs  = final["I"]
    q_mfs  = final["Q"]
    u_mfs  = final["U"]
    p_mfs  = np.sqrt(q_mfs ** 2 + u_mfs ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        fp_mfs = np.where(i_mfs > 0, p_mfs / i_mfs, np.nan)

    products = [
        (i_mfs,  "I MFS residual",         _CMAP_I,  None),
        (q_mfs,  "Q MFS residual",         _CMAP_QU, None),
        (u_mfs,  "U MFS residual",         _CMAP_QU, None),
        (p_mfs,  "P MFS residual",         _CMAP_P,  None),
        (fp_mfs, "Fractional pol P/I",     "viridis", (0, 1)),
    ]

    vroom_d = diag.get("vroom_diag", {})
    rm_val  = None
    if vroom_d.get("result") is not None:
        res = vroom_d["result"]
        if res.components:
            rm_val = res.components[0].rm_mean

    # 2-row × 3-col grid
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "APCLEAN — Final MFS Products  "
        + (f"(VROOM RM={rm_val:.1f} rad/m²)" if rm_val is not None else ""),
        fontsize=12, fontweight="bold",
    )

    for idx, (arr, title, cmap, vlim) in enumerate(products):
        r, c = divmod(idx, 3)
        ax   = fig.add_subplot(gs[r, c])
        vmax = float(np.nanmax(np.abs(arr)))
        if vlim:
            vmin_p, vmax_p = vlim
        elif cmap == _CMAP_QU:
            vmin_p, vmax_p = -vmax, vmax
        else:
            vmin_p, vmax_p = 0, vmax
        im = ax.imshow(arr, origin="lower", cmap=cmap,
                       vmin=vmin_p, vmax=vmax_p, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        _add_colorbar(ax, im)

    # 6th panel: parameter text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    params = diag.get("params", {})
    sl     = diag.get("sidelobe_level", float("nan"))
    lines  = [
        f"MS:        {params.get('ms', '?')}",
        f"imsize:    {params.get('imsize', '?')}",
        f"cell:      {params.get('cell', '?')}",
        f"loop_gain: {params.get('loop_gain', '?')}",
        f"cyclefactor: {params.get('cyclefactor', '?')}",
        f"threshold:  {params.get('threshold_sigma', '?')}σ",
        f"p_thresh:  {params.get('p_snr_threshold', '?')}σ",
        f"sidelobe:  {sl:.4f}",
        f"nchans:    {len(freq_hz) if freq_hz is not None else '?'}",
    ]
    ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes,
             fontsize=8, va="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax6.set_title("Run parameters", fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def _page_vroom_spectra(
    pdf    : PdfPages,
    diag   : dict,
    freq_hz: np.ndarray,
) -> None:
    vroom_d = diag.get("vroom_diag", {})
    q_norm  = vroom_d.get("q_norm")
    u_norm  = vroom_d.get("u_norm")
    if q_norm is None or u_norm is None:
        return

    dirty_i = diag.get("initial_dirty_i")
    dirty_q = diag.get("initial_dirty_q")
    dirty_u = diag.get("initial_dirty_u")

    nu_ghz  = freq_hz / 1e9 if freq_hz is not None else np.arange(len(q_norm))
    lsq     = (3e8 / freq_hz) ** 2 if freq_hz is not None else np.arange(len(q_norm))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("VROOM-SBI: Faraday spectrum at source pixel",
                 fontsize=12, fontweight="bold")

    # dirty Q/I and U/I vs VROOM model
    with np.errstate(invalid="ignore", divide="ignore"):
        qi_dirty = (np.where(dirty_i > 0, dirty_q / dirty_i, np.nan)
                    if dirty_i is not None else None)
        ui_dirty = (np.where(dirty_i > 0, dirty_u / dirty_i, np.nan)
                    if dirty_i is not None else None)

    # Top left: Q/I
    ax = axes[0, 0]
    if qi_dirty is not None:
        ax.plot(nu_ghz, qi_dirty, ".", color="grey",  ms=3, label="Dirty Q/I")
    ax.plot(nu_ghz, q_norm, "-",    color="royalblue",      lw=1.5, label="VROOM Q/I model")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Q / I")
    ax.set_title("Q/I spectrum")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3)

    # Top right: U/I
    ax = axes[0, 1]
    if ui_dirty is not None:
        ax.plot(nu_ghz, ui_dirty, ".", color="grey",  ms=3, label="Dirty U/I")
    ax.plot(nu_ghz, u_norm, "-",    color="tomato",          lw=1.5, label="VROOM U/I model")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("U / I")
    ax.set_title("U/I spectrum")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3)

    # Bottom left: P = sqrt(Q²+U²) vs model
    ax = axes[1, 0]
    p_model = np.sqrt(q_norm ** 2 + u_norm ** 2)
    if qi_dirty is not None and ui_dirty is not None:
        p_dirty = np.sqrt(np.nan_to_num(qi_dirty) ** 2
                          + np.nan_to_num(ui_dirty) ** 2)
        ax.plot(nu_ghz, p_dirty, ".", color="grey", ms=3, label="Dirty P/I")
    ax.plot(nu_ghz, p_model, "-", color="purple", lw=1.5, label="VROOM P/I model")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("P / I")
    ax.set_title("Polarized fraction vs frequency")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3)

    # Bottom right: PA = 0.5 atan2(U/I, Q/I)
    ax = axes[1, 1]
    pa_model = 0.5 * np.degrees(np.arctan2(u_norm, q_norm))
    if qi_dirty is not None and ui_dirty is not None:
        pa_dirty = 0.5 * np.degrees(
            np.arctan2(np.nan_to_num(ui_dirty), np.nan_to_num(qi_dirty))
        )
        ax.plot(lsq, pa_dirty, ".", color="grey", ms=3, label="Dirty PA")
    ax.plot(lsq, pa_model, "-", color="darkorange", lw=1.5, label="VROOM PA model")
    ax.set_xlabel("λ² (m²)")
    ax.set_ylabel("PA (°)")
    ax.set_title("PA vs λ² (Faraday rotation)")
    ax.legend(fontsize=8)
    ax.grid(linewidth=0.3)

    # Annotate VROOM result
    result = vroom_d.get("result")
    pixel  = vroom_d.get("pixel")
    if result is not None and result.components:
        comp = result.components[0]
        txt  = (
            f"pixel {pixel}   model={result.model_type}\n"
            f"N_comp={result.n_components}\n"
            f"RM={comp.rm_mean:.2f} rad/m²\n"
            f"P_amp={np.sqrt(comp.q_mean**2+comp.u_mean**2):.4f}"
        )
        fig.text(0.5, 0.01, txt, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)


def _page_minor_traces(pdf: PdfPages, diag: dict) -> None:
    cycles = diag.get("major_cycles", [])
    if not cycles:
        return

    n  = len(cycles)
    nc = min(n, 3)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr), squeeze=False)
    fig.suptitle("Minor-cycle traces (peak vs iteration per major cycle)",
                 fontsize=12, fontweight="bold")

    for idx, cyc in enumerate(cycles):
        r, c = divmod(idx, nc)
        ax   = axes[r, c]
        trace = cyc.get("minor_trace", [])
        if trace:
            iters = [t[0] for t in trace]
            peaks = [t[1] for t in trace]
            ax.semilogy(iters, peaks, "-", color="royalblue", linewidth=1)
        ax.axhline(cyc["cycle_thresh"],  color="orange",  linestyle="--",
                   linewidth=0.8, label=f"cycle_thresh")
        ax.axhline(cyc["global_thresh"], color="crimson",  linestyle=":",
                   linewidth=0.8, label=f"global_thresh")
        ax.set_title(f"Major cycle {idx + 1}  ({cyc['n_clean_iters']} iters)",
                     fontsize=9)
        ax.set_xlabel("Minor iteration")
        ax.set_ylabel("I peak (Jy/beam)")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", linewidth=0.3)

    for idx in range(n, nr * nc):
        r, c = divmod(idx, nc)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================================
# Utility
# ============================================================================

def _add_colorbar(ax, im) -> None:
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="4%", pad=0.03)
    plt.colorbar(im, cax=cax)
