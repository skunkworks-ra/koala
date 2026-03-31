"""
APCLEAN — Amortized Polarization CLEAN
=======================================
Major cycle : casatasks.tclean(niter=0)
Minor cycle : VROOM-SBI (polarized pixels) + Hogbom (I-only pixels)

Quick start
-----------
    from apclean import APClean

    cleaner = APClean(
        ms            = "obs.ms",
        imagename     = "apclean_work",
        imsize        = 256,
        cell          = "15arcsec",
        vroom_config  = "config.yaml",
        vroom_model_dir = "models/",
    )
    cleaner.run(output_prefix="apclean_out", debug=True)
"""

from apclean.cleaner import APClean

__all__ = ["APClean"]
__version__ = "0.1.0"
