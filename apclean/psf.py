"""
apclean.psf
===========
PSF utilities: boundary-safe subtraction, sidelobe level, clean beam fitting.
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def subtract_psf_at(
    residual  : np.ndarray,
    psf       : np.ndarray,
    y         : int,
    x         : int,
    amplitude : float,
) -> None:
    """
    Subtract amplitude × PSF centred at (y, x) from *residual* in-place.

    Both arrays are 2-D (single channel).  PSF peak is assumed at (ny//2, nx//2).
    Boundary-safe: clips overlap region so no circular wrap occurs.
    """
    ny, nx = residual.shape
    py, px = psf.shape
    cy, cx = py // 2, px // 2

    r_y0 = max(0, y - cy);  r_y1 = min(ny, y - cy + py)
    r_x0 = max(0, x - cx);  r_x1 = min(nx, x - cx + px)
    p_y0 = r_y0 - (y - cy); p_y1 = p_y0 + (r_y1 - r_y0)
    p_x0 = r_x0 - (x - cx); p_x1 = p_x0 + (r_x1 - r_x0)

    residual[r_y0:r_y1, r_x0:r_x1] -= amplitude * psf[p_y0:p_y1, p_x0:p_x1]


def compute_sidelobe_level(
    psf_cube           : np.ndarray,
    sidelobe_radius_pix: int = 15,
) -> float:
    """
    Estimate PSF sidelobe level = max|PSF| / PSF_peak outside the main lobe.

    Parameters
    ----------
    psf_cube            : (nfreq, ndec, nra)
    sidelobe_radius_pix : pixels around centre excluded (should cover main lobe)

    Returns
    -------
    sidelobe_level : float in (0, 1)
    """
    psf_mean = np.nanmean(psf_cube, axis=0)
    peak     = float(psf_mean.max())
    if peak <= 0:
        return 0.1

    ny, nx = psf_mean.shape
    yy, xx = np.ogrid[:ny, :nx]
    r       = np.sqrt((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2)
    outside = r > sidelobe_radius_pix
    level   = float(np.max(np.abs(psf_mean[outside]))) / peak
    logger.info(
        f"PSF sidelobe level: {level:.4f}  (radius={sidelobe_radius_pix} pix)"
    )
    return level


def fit_clean_beam(psf_cube: np.ndarray) -> tuple[float, float, float]:
    """
    Fit a rotated 2-D Gaussian to the central peak of the channel-averaged PSF.

    Returns
    -------
    bmaj_pix : float   FWHM of major axis in pixels
    bmin_pix : float   FWHM of minor axis in pixels
    bpa_deg  : float   Position angle (degrees, N through E)
    """
    psf_mean = np.mean(psf_cube, axis=0)
    ny, nx   = psf_mean.shape
    cy, cx   = ny // 2, nx // 2
    hw       = 20
    patch    = psf_mean[max(0, cy - hw): cy + hw + 1,
                        max(0, cx - hw): cx + hw + 1]
    ph, pw   = patch.shape
    yy, xx   = np.mgrid[0:ph, 0:pw].astype(float)
    yy -= ph // 2;  xx -= pw // 2

    def _g(coords, amp, sx, sy, th):
        xi, yi = coords
        ct, st = np.cos(th), np.sin(th)
        xr =  ct * xi + st * yi
        yr = -st * xi + ct * yi
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    try:
        popt, _ = curve_fit(
            _g, (xx.ravel(), yy.ravel()), patch.ravel(),
            p0    = [1., 3., 2., 0.],
            bounds= ([0, .5, .5, -np.pi / 2], [2., 30., 30., np.pi / 2]),
            maxfev= 10000,
        )
        _, sx, sy, th = popt
    except Exception as exc:
        logger.warning(f"Clean beam fit failed ({exc}) — using defaults.")
        sx, sy, th = 3., 2., 0.

    fwhm_x  = 2.3548 * abs(sx)
    fwhm_y  = 2.3548 * abs(sy)
    bmaj    = max(fwhm_x, fwhm_y)
    bmin    = min(fwhm_x, fwhm_y)
    bpa_deg = np.degrees(th) + (90. if fwhm_y > fwhm_x else 0.)
    logger.info(f"Clean beam: {bmaj:.2f}×{bmin:.2f} pix  PA={bpa_deg:.1f}°")
    return bmaj, bmin, bpa_deg


def restore_cube(
    model   : np.ndarray,
    residual: np.ndarray,
    psf_cube: np.ndarray,
) -> np.ndarray:
    """
    Restored image = model convolved with clean beam + residual.

    Parameters
    ----------
    model, residual : (nfreq, ndec, nra)
    psf_cube        : (nfreq, ndec, nra)  — used for beam fitting

    Returns
    -------
    restored : (nfreq, ndec, nra)
    """
    bmaj, bmin, bpa = fit_clean_beam(psf_cube)
    ksize    = int(np.ceil(bmaj * 4)) | 1
    beam     = Gaussian2DKernel(
        x_stddev = bmin / 2.3548,
        y_stddev = bmaj / 2.3548,
        theta    = np.deg2rad(bpa),
        x_size   = ksize,
        y_size   = ksize,
    )
    # Normalise to peak=1 (CLEAN Jy/beam convention, not Jy)
    beam_arr = beam.array / beam.array.max()

    nfreq     = model.shape[0]
    restored  = np.empty_like(model)
    for c in range(nfreq):
        restored[c] = (
            convolve_fft(model[c], beam_arr, normalize_kernel=False)
            + residual[c]
        )
    return restored
