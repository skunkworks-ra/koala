"""
Koala — Wideband Spectral + Polarimetric CLEAN
===============================================
Major cycle : casatasks.tclean(niter=0)
Minor cycle : VROOM-SBI spectral shape (all pixels) + Faraday rotation (polarized)

Modes
-----
  spectra     : Stokes I only — spectral shape inference per point source
  spectra+pol : Full IQUV — spectral shape I + Faraday Q/U inference

Quick start
-----------
    from koala import Koala

    cleaner = Koala(
        ms              = "obs.ms",
        imagename       = "koala_work",
        imsize          = 256,
        cell            = "15arcsec",
        vroom_model_dir = "models/",
        mode            = "spectra+pol",
    )
    cleaner.run(output_prefix="koala_out", debug=True)
"""

from koala.cleaner import Koala

__all__ = ["Koala"]
__version__ = "0.2.0"
