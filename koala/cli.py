"""
koala.cli
=========
Command-line interface for Koala.

Usage
-----
    koala --ms obs.ms --imagename work/koala \\
          --imsize 256 --cell 15arcsec \\
          --vroom-model-dir models/ \\
          --mode spectra+pol --debug
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="koala",
        description="Koala — wideband spectral + polarimetric CLEAN (VROOM-SBI + tclean)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument("--ms",        required=True,
                     help="Measurement set path.")
    req.add_argument("--imagename", required=True,
                     help="tclean working image prefix (e.g. work/koala).")
    req.add_argument("--imsize",    required=True, type=int,
                     help="Square image size in pixels.")
    req.add_argument("--cell",      required=True,
                     help="Cell size string (e.g. '15arcsec').")

    # ── VROOM-SBI ─────────────────────────────────────────────────────────
    vroom = p.add_argument_group("VROOM-SBI")
    vroom.add_argument("--vroom-model-dir", default="models",
                       help="Path to trained VROOM-SBI models directory.")
    vroom.add_argument("--n-samples",       type=int, default=1000,
                       help="Number of VROOM posterior samples per pixel.")
    vroom.add_argument("--device",          default="cuda",
                       choices=["cuda", "cpu"],
                       help="PyTorch device for VROOM inference.")

    # ── mode ──────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="spectra+pol",
                   choices=["spectra", "spectra+pol"],
                   help="'spectra' = Stokes I only with spectral shape SED. "
                        "'spectra+pol' = full IQUV with spectral I + Faraday Q/U.")

    # ── imaging ───────────────────────────────────────────────────────────
    img = p.add_argument_group("imaging")
    img.add_argument("--reffreq",  default="1.5GHz",
                     help="Reference frequency (e.g. '1.5GHz').")
    img.add_argument("--robust",   type=float, default=0.5,
                     help="Briggs robust weighting parameter.")

    # ── CLEAN ─────────────────────────────────────────────────────────────
    cln = p.add_argument_group("CLEAN")
    cln.add_argument("--loop-gain",       type=float, default=0.1,
                     help="Minor-cycle loop gain.")
    cln.add_argument("--threshold-sigma", type=float, default=3.0,
                     help="Global stopping threshold in units of σ_I.")
    cln.add_argument("--p-snr-threshold", type=float, default=5.0,
                     help="Polarization detection threshold in units of σ_P.")
    cln.add_argument("--cyclefactor",     type=float, default=1.5,
                     help="Major-cycle trigger: minor cycle stops when peak < "
                          "cyclefactor × sidelobe_level × P0.")
    cln.add_argument("--sidelobe-radius", type=int,   default=15,
                     help="Pixels around PSF peak excluded from sidelobe "
                          "estimation (should cover main lobe).")
    cln.add_argument("--niter-minor-max", type=int,   default=5000,
                     help="Hard cap on minor-cycle CLEAN iterations per "
                          "major cycle.")
    cln.add_argument("--max-major",       type=int,   default=10,
                     help="Maximum number of major cycles.")

    # ── output ────────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument("--output-prefix", default="koala_out",
                     help="Prefix for all output FITS files and the run log.")
    out.add_argument("--debug", action="store_true",
                     help="Generate multi-page diagnostic PDF after run.")
    out.add_argument("--log-level",
                     default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Python logging level.")

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.output_prefix}.log"),
        ],
    )

    from koala import Koala

    cleaner = Koala(
        ms              = args.ms,
        imagename       = args.imagename,
        imsize          = args.imsize,
        cell            = args.cell,
        reffreq         = args.reffreq,
        robust          = args.robust,
        vroom_model_dir = args.vroom_model_dir,
        mode            = args.mode,
        loop_gain       = args.loop_gain,
        threshold_sigma = args.threshold_sigma,
        p_snr_threshold = args.p_snr_threshold,
        cyclefactor     = args.cyclefactor,
        sidelobe_radius = args.sidelobe_radius,
        niter_minor_max = args.niter_minor_max,
        max_major       = args.max_major,
        n_samples       = args.n_samples,
        device          = args.device,
    )

    cleaner.run(output_prefix=args.output_prefix, debug=args.debug)


if __name__ == "__main__":
    main()
