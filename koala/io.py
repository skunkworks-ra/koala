"""
koala.io
========
CASA image I/O via casatools and FITS output via astropy.
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CASA image reading
# ---------------------------------------------------------------------------

def read_casa_iquv(path: str) -> tuple[dict[str, np.ndarray], np.ndarray, WCS]:
    """
    Read a CASA image that may have one or more Stokes planes.

    Returns
    -------
    planes  : dict  {Stokes letter: (nfreq, ndec, nra) float64}
    freq_hz : ndarray (nfreq,)
    wcs_2d  : astropy WCS, 2-D celestial
    """
    from casatools import image as iatool

    ia = iatool()
    ia.open(str(path))
    # getchunk → (nra, ndec, nstokes, nfreq)  CASA native order
    raw  = np.array(ia.getchunk(), dtype=np.float64)
    csys = ia.coordsys().torecord()
    ia.close()

    nra, ndec, nstokes, nfreq = raw.shape

    freq_hz = _extract_freq(csys, nfreq)
    wcs_2d  = _extract_wcs2d(csys)
    names   = _stokes_names(csys)

    planes = {
        s: raw[:, :, i, :].transpose(2, 1, 0).copy()  # → (nfreq, ndec, nra)
        for i, s in enumerate(names)
    }
    return planes, freq_hz, wcs_2d


def _extract_freq(csys: dict, nfreq: int) -> np.ndarray:
    spec  = csys.get("spectral2", csys.get("spectral1", {})).get("wcs", {})
    crval = float(spec["crval"])
    cdelt = float(spec["cdelt"])
    crpix = float(spec["crpix"])
    return crval + (np.arange(nfreq) - crpix) * cdelt


def _extract_wcs2d(csys: dict) -> WCS:
    d   = csys.get("direction0", {})
    wcs = WCS(naxis=2)
    if d:
        wcs.wcs.crpix = [d["crpix"][0] + 1, d["crpix"][1] + 1]
        wcs.wcs.cdelt = np.degrees(d["cdelt"]).tolist()
        wcs.wcs.crval = np.degrees(d["crval"]).tolist()
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return wcs


def _stokes_names(csys: dict) -> list[str]:
    for key in ("stokes1", "stokes2"):
        if key in csys:
            return list(csys[key]["stokes"])
    return ["I"]


# ---------------------------------------------------------------------------
# CASA model image writing
# ---------------------------------------------------------------------------

def write_model_to_casa(
    path          : str,
    i_model       : np.ndarray,
    q_model       : np.ndarray,
    u_model       : np.ndarray,
    template_path : str = "",
) -> None:
    """
    Write I/Q/U model arrays into a CASA image at *path*.

    If the image does not exist it is created by cloning *template_path*
    (the residual image) and zeroing the data.
    """
    from casatools import image as iatool

    if not os.path.exists(str(path)):
        if not template_path:
            raise RuntimeError(
                f"Model image '{path}' does not exist and no template supplied."
            )
        shutil.copytree(str(template_path), str(path))
        ia_tmp = iatool()
        ia_tmp.open(str(path))
        ia_tmp.putchunk(np.zeros_like(ia_tmp.getchunk()))
        ia_tmp.close()
        logger.info(f"Created model CASA image: {path}")

    ia = iatool()
    ia.open(str(path))
    data = np.array(ia.getchunk())   # (nra, ndec, nstokes, nfreq)

    def _to_casa(arr: np.ndarray) -> np.ndarray:
        return arr.transpose(2, 1, 0).astype(np.float64)

    data[:, :, 0, :] = _to_casa(i_model)
    if data.shape[2] > 1:
        data[:, :, 1, :] = _to_casa(q_model)
    if data.shape[2] > 2:
        data[:, :, 2, :] = _to_casa(u_model)

    ia.putchunk(data)
    ia.close()


# ---------------------------------------------------------------------------
# FITS output helpers
# ---------------------------------------------------------------------------

def write_fits_cube(
    data    : np.ndarray,
    wcs_2d  : WCS,
    freq_hz : np.ndarray,
    path    : str,
) -> None:
    """Write a (nfreq, ndec, nra) array as a 3-D FITS cube."""
    hdr = wcs_2d.to_header()
    hdr.update({
        "NAXIS" : 3,
        "NAXIS3": len(freq_hz),
        "CTYPE3": "FREQ",
        "CRPIX3": 1.0,
        "CRVAL3": float(freq_hz[0]),
        "CDELT3": float(freq_hz[1] - freq_hz[0]) if len(freq_hz) > 1 else 1.0,
        "CUNIT3": "Hz",
        "BUNIT" : "Jy/beam",
    })
    fits.writeto(path, data.astype(np.float32), header=hdr, overwrite=True)
    logger.debug(f"Wrote {path}")


def write_fits_map(data: np.ndarray, wcs_2d: WCS, path: str) -> None:
    """Write a 2-D array as a FITS image."""
    hdr = wcs_2d.to_header()
    fits.writeto(path, data.astype(np.float32), header=hdr, overwrite=True)
    logger.debug(f"Wrote {path}")
