"""
apclean.cleaner
===============
APClean — main orchestrator.

Major cycle : casatasks.tclean(niter=0)
Minor cycle : MinorCycle (VROOM-SBI + Hogbom)
Convergence : cycle-factor method  (cyclefactor × sidelobe × P0)
"""

from __future__ import annotations

import logging

import numpy as np
from astropy.wcs import WCS

from apclean.io import (
    read_casa_iquv,
    write_fits_cube,
    write_fits_map,
    write_model_to_casa,
)
from apclean.minor_cycle import MinorCycle
from apclean.psf import compute_sidelobe_level, restore_cube

logger = logging.getLogger(__name__)


class APClean:
    """
    Amortized Polarization CLEAN.

    Parameters
    ----------
    ms               : str    Measurement set path.
    imagename        : str    tclean working image prefix.
    imsize           : int    Square image size in pixels.
    cell             : str    Cell size string, e.g. '15arcsec'.
    reffreq          : str    Reference frequency, e.g. '1.5GHz'.
    robust           : float  Briggs robust weighting (default 0.5).
    vroom_config     : str    Path to VROOM-SBI config.yaml.
    vroom_model_dir  : str    Path to trained VROOM-SBI models directory.
    loop_gain        : float  Minor-cycle loop gain (default 0.1).
    threshold_sigma  : float  Global stopping threshold in sigma_I (default 3).
    p_snr_threshold  : float  Polarization detection in sigma_P (default 5).
    cyclefactor      : float  Major-cycle trigger: stop minor cycle when
                              peak < cyclefactor × sidelobe_level × P0
                              (default 1.5, matches tclean convention).
    sidelobe_radius  : int    Pixels around PSF peak excluded for sidelobe
                              estimation (default 15).
    niter_minor_max  : int    Hard cap on minor-cycle iterations (default 5000).
    max_major        : int    Maximum major cycles (default 10).
    n_samples        : int    VROOM-SBI posterior samples (default 1000).
    device           : str    VROOM device 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        ms              : str,
        imagename       : str,
        imsize          : int,
        cell            : str,
        reffreq         : str   = "1.5GHz",
        robust          : float = 0.5,
        vroom_model_dir : str   = "models",
        loop_gain       : float = 0.1,
        threshold_sigma : float = 3.0,
        p_snr_threshold : float = 5.0,
        cyclefactor     : float = 1.5,
        sidelobe_radius : int   = 15,
        niter_minor_max : int   = 5000,
        max_major       : int   = 10,
        n_samples       : int   = 1000,
        device          : str   = "cuda",
    ):
        self.ms              = str(ms)
        self.imagename       = str(imagename)
        self.imsize          = imsize
        self.cell            = cell
        self.reffreq         = reffreq
        self.robust          = robust
        self.loop_gain       = loop_gain
        self.threshold_sigma = threshold_sigma
        self.p_snr_threshold = p_snr_threshold
        self.cyclefactor     = cyclefactor
        self.sidelobe_radius = sidelobe_radius
        self.niter_minor_max = niter_minor_max
        self.max_major       = max_major
        self.n_samples       = n_samples

        # Populated after first major cycle
        self.freq_hz  : np.ndarray | None = None
        self.wcs_2d   : WCS | None        = None
        self.n_freq = self.n_dec = self.n_ra = 0
        self.sidelobe_level : float = 0.1

        # Working arrays  (nfreq, ndec, nra)
        self.i_residual = self.q_residual = self.u_residual = None
        self.i_psf      = None
        self.i_model    = self.q_model = self.u_model = None
        self.sigma_i    : np.ndarray | None = None
        self.mean_sigma_i : float = 0.0
        self.p_map      : np.ndarray | None = None
        self.sigma_p    : float = 0.0

        # VROOM parameter maps  (ndec, nra) — filled per polarized pixel
        self.rm_map        : np.ndarray | None = None
        self.p_amp_map     : np.ndarray | None = None
        self.chi0_map      : np.ndarray | None = None
        self.n_comp_map    : np.ndarray | None = None
        self.sigma_phi_map : np.ndarray | None = None
        self.delta_phi_map : np.ndarray | None = None

        # Diagnostics (filled during run, used by diagnostics.py)
        self.diag : dict = {
            "params"            : self._param_dict(),
            "sidelobe_level"    : None,
            "major_cycles"      : [],
            "mfs_snapshots"     : [],   # list of {I, Q, U} dicts
            "source_pixel"      : None,
            "initial_dirty_i"   : None,
            "initial_dirty_q"   : None,
            "initial_dirty_u"   : None,
            "vroom_diag"        : {"q_norm": None, "u_norm": None,
                                   "result": None,  "pixel": None},
        }

        # Load VROOM-SBI engine
        logger.info("Loading VROOM-SBI engine …")
        from src.inference import InferenceEngine
        from src.simulator.physics import freq_to_lambda_sq

        self._freq_to_lambda_sq = freq_to_lambda_sq
        self._engine = InferenceEngine(
            config=None, model_dir=vroom_model_dir, device=device
        )
        self._engine.load_models()
        logger.info("APClean ready.")

    def _param_dict(self) -> dict:
        return {k: getattr(self, k, None) for k in (
            "ms", "imagename", "imsize", "cell", "reffreq", "robust",
            "vroom_model_dir", "loop_gain", "threshold_sigma", "p_snr_threshold",
            "cyclefactor", "sidelobe_radius", "niter_minor_max", "max_major",
        )}

    # ------------------------------------------------------------------
    # Major cycle
    # ------------------------------------------------------------------

    def _run_major(self, first: bool) -> None:
        from casatasks import tclean
        kw = dict(
            vis         = self.ms,
            imagename   = self.imagename,
            imsize      = [self.imsize, self.imsize],
            cell        = [self.cell],
            stokes      = "IQUV",
            specmode    = "cube",
            nchan       = -1,
            reffreq     = self.reffreq,
            weighting   = "briggs",
            robust      = self.robust,
            niter       = 0,
            deconvolver = "hogbom",
            restoration = False,
        )
        if first:
            logger.info("Major cycle 1: computing PSF + dirty image …")
            tclean(**kw, calcpsf=True, calcres=True)
        else:
            logger.info("Major cycle: re-gridding with updated model …")
            tclean(**kw, calcpsf=False, calcres=True)

    # ------------------------------------------------------------------
    # Read residual and PSF from CASA images
    # ------------------------------------------------------------------

    def _update_from_casa(self) -> None:
        res_planes, freq_hz, wcs_2d = read_casa_iquv(
            f"{self.imagename}.residual"
        )

        if self.freq_hz is None:
            self.freq_hz  = freq_hz
            self.wcs_2d   = wcs_2d
            self.n_freq   = len(freq_hz)
            self.n_dec    = res_planes["I"].shape[1]
            self.n_ra     = res_planes["I"].shape[2]
            logger.info(
                f"Grid: {self.n_freq} channels  {self.n_dec}×{self.n_ra} pixels"
            )

            shape3 = (self.n_freq, self.n_dec, self.n_ra)
            shape2 = (self.n_dec, self.n_ra)

            self.i_model       = np.zeros(shape3)
            self.q_model       = np.zeros(shape3)
            self.u_model       = np.zeros(shape3)
            self.rm_map        = np.full(shape2, np.nan)
            self.p_amp_map     = np.full(shape2, np.nan)
            self.chi0_map      = np.full(shape2, np.nan)
            self.n_comp_map    = np.full(shape2, np.nan)
            self.sigma_phi_map = np.full(shape2, np.nan)
            self.delta_phi_map = np.full(shape2, np.nan)

            psf_planes, _, _ = read_casa_iquv(f"{self.imagename}.psf")
            self.i_psf = psf_planes["I"]

            self.sidelobe_level = compute_sidelobe_level(
                self.i_psf, self.sidelobe_radius
            )
            self.diag["sidelobe_level"] = self.sidelobe_level

            # Frequency matching for VROOM
            cube_lsq = self._freq_to_lambda_sq(freq_hz)
            self._cube_lambda_sq = cube_lsq
            self._freq_sort_idx  = self._match_frequencies(freq_hz)

        self.i_residual = res_planes["I"]
        self.q_residual = res_planes.get("Q", np.zeros_like(self.i_residual))
        self.u_residual = res_planes.get("U", np.zeros_like(self.i_residual))

        logger.info(
            f"  Residual I peak = {np.nanmax(np.abs(self.i_residual)):.4f} Jy/beam"
        )

    def _match_frequencies(self, freq_hz: np.ndarray) -> np.ndarray | None:
        if not self._engine.model_lambda_sq:
            return None
        from src.simulator.physics import freq_to_lambda_sq
        ref_lsq  = next(iter(self._engine.model_lambda_sq.values()))
        cube_lsq = freq_to_lambda_sq(freq_hz)
        n        = len(cube_lsq)
        if len(ref_lsq) != n:
            logger.warning("Channel count mismatch between cube and VROOM model.")
            return None
        if np.allclose(cube_lsq, ref_lsq, rtol=1e-2):
            return None
        if np.allclose(cube_lsq[::-1], ref_lsq, rtol=1e-2):
            logger.info("Frequencies reversed — reordering for VROOM.")
            return np.arange(n)[::-1]
        logger.warning("Frequency grid mismatch — VROOM RM values may be unreliable.")
        return None

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------

    def _estimate_noise(self) -> None:
        sigma_i = np.zeros(self.n_freq)
        for c in range(self.n_freq):
            ch  = self.i_residual[c]
            med = np.nanmedian(ch)
            sigma_i[c] = 1.4826 * np.nanmedian(np.abs(ch - med))
        self.sigma_i      = sigma_i
        self.mean_sigma_i = float(np.nanmean(sigma_i))

        p_map        = np.nanmean(
            np.sqrt(self.q_residual ** 2 + self.u_residual ** 2), axis=0
        )
        self.p_map   = p_map
        p_med        = np.nanmedian(p_map)
        self.sigma_p = 1.4826 * float(np.nanmedian(np.abs(p_map - p_med)))

        logger.info(
            f"  sigma_I={self.mean_sigma_i:.4f}  sigma_P={self.sigma_p:.4f}  "
            f"P_thresh({self.p_snr_threshold}σ)="
            f"{self.p_snr_threshold * self.sigma_p:.4f}"
        )

    # ------------------------------------------------------------------
    # Store VROOM parameter maps from minor cycle result
    # ------------------------------------------------------------------

    def _store_vroom_maps(self, vroom_diag: dict) -> None:
        result = vroom_diag.get("result")
        pixel  = vroom_diag.get("pixel")
        if result is None or pixel is None:
            return
        y, x = pixel
        self.n_comp_map[y, x] = result.n_components
        if result.components:
            comp = result.components[0]
            self.rm_map[y, x]    = comp.rm_mean
            self.p_amp_map[y, x] = np.sqrt(comp.q_mean ** 2 + comp.u_mean ** 2)
            if comp.chi0_mean is not None:
                self.chi0_map[y, x] = comp.chi0_mean
            if comp.sigma_phi_mean is not None:
                self.sigma_phi_map[y, x] = comp.sigma_phi_mean
            if comp.delta_phi_mean is not None:
                self.delta_phi_map[y, x] = comp.delta_phi_mean

    # ------------------------------------------------------------------
    # MFS collapse (σ_I-weighted channel mean)
    # ------------------------------------------------------------------

    def _collapse_mfs(
        self,
        i_cube: np.ndarray,
        q_cube: np.ndarray,
        u_cube: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = self.sigma_i > 0
        w     = np.where(valid, 1.0 / self.sigma_i ** 2, 0.0)
        wsum  = w.sum()
        return (
            np.einsum("c,cyx->yx", w, i_cube) / wsum,
            np.einsum("c,cyx->yx", w, q_cube) / wsum,
            np.einsum("c,cyx->yx", w, u_cube) / wsum,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, output_prefix: str = "apclean", debug: bool = False) -> None:
        """
        Run all major+minor cycles, save FITS outputs, optionally plot diagnostics.

        Parameters
        ----------
        output_prefix : str   Prefix for all output files.
        debug         : bool  If True, generate diagnostic PDF after run.
        """
        log_path = f"{output_prefix}.log"
        log_fh   = open(log_path, "w")
        log_fh.write(
            f"APCLEAN  MS={self.ms}\n"
            f"cyclefactor={self.cyclefactor}  sidelobe_radius={self.sidelobe_radius}\n"
            f"loop_gain={self.loop_gain}  threshold={self.threshold_sigma}σ  "
            f"p_threshold={self.p_snr_threshold}σ\n"
            f"{'iter':<8}{'y':<5}{'x':<5}{'peak':<14}{'pol':<5}model\n"
            + "-" * 70 + "\n"
        )

        minor = MinorCycle(
            engine          = self._engine,
            freq_hz         = self.freq_hz if self.freq_hz is not None else np.array([]),
            cube_lambda_sq  = getattr(self, "_cube_lambda_sq", np.array([])),
            freq_sort_idx   = getattr(self, "_freq_sort_idx", None),
            loop_gain       = self.loop_gain,
            p_snr_threshold = self.p_snr_threshold,
            n_samples       = self.n_samples,
        )

        global_converged = False

        for major_idx in range(1, self.max_major + 1):
            logger.info(f"=== Major cycle {major_idx}/{self.max_major} ===")
            log_fh.write(f"\n--- Major cycle {major_idx} ---\n")

            # ---- Major cycle: tclean predicts model, subtracts, grids ----
            self._run_major(first=(major_idx == 1))
            self._update_from_casa()
            self._estimate_noise()

            # MinorCycle needs freq_hz and lambda_sq — set after first update
            if major_idx == 1:
                minor.freq_hz        = self.freq_hz
                minor.cube_lambda_sq = self._cube_lambda_sq
                minor.freq_sort_idx  = self._freq_sort_idx
                minor.n_freq         = self.n_freq

            # MFS snapshot for diagnostics
            self.diag["mfs_snapshots"].append({
                "I": np.nanmean(self.i_residual, axis=0).copy(),
                "Q": np.nanmean(self.q_residual, axis=0).copy(),
                "U": np.nanmean(self.u_residual, axis=0).copy(),
            })

            # Save initial dirty spectra at source pixel (once)
            if major_idx == 1:
                m      = np.nanmean(np.abs(self.i_residual), axis=0)
                y_src, x_src = np.unravel_index(np.nanargmax(m), m.shape)
                self.diag["source_pixel"]   = (int(y_src), int(x_src))
                self.diag["initial_dirty_i"] = self.i_residual[:, y_src, x_src].copy()
                self.diag["initial_dirty_q"] = self.q_residual[:, y_src, x_src].copy()
                self.diag["initial_dirty_u"] = self.u_residual[:, y_src, x_src].copy()

            global_thresh = self.threshold_sigma * self.mean_sigma_i
            peak_before   = float(np.nanmax(np.abs(self.i_residual)))

            if major_idx > 1 and peak_before < global_thresh:
                logger.info("Peak already below threshold — done.")
                global_converged = True
                break

            p0           = peak_before
            cycle_thresh = self.cyclefactor * self.sidelobe_level * p0
            logger.info(
                f"  P0={p0:.4f}  cycle_thresh={cycle_thresh:.4f}  "
                f"global_thresh={global_thresh:.4f}"
            )

            # ---- Minor cycle ----
            vroom_diag_this = {"q_norm": None, "u_norm": None,
                               "result": None,  "pixel": None}
            n_iters, trace, status = minor.run(
                i_residual   = self.i_residual,
                q_residual   = self.q_residual,
                u_residual   = self.u_residual,
                i_model      = self.i_model,
                q_model      = self.q_model,
                u_model      = self.u_model,
                i_psf        = self.i_psf,
                p_map        = self.p_map,
                sigma_p      = self.sigma_p,
                cycle_thresh = cycle_thresh,
                global_thresh= global_thresh,
                niter_max    = self.niter_minor_max,
                log_fh       = log_fh,
                vroom_diag   = vroom_diag_this,
            )

            # Store VROOM maps and diagnostics
            self._store_vroom_maps(vroom_diag_this)
            if (self.diag["vroom_diag"]["q_norm"] is None
                    and vroom_diag_this["q_norm"] is not None):
                self.diag["vroom_diag"] = vroom_diag_this

            _, _, peak_after = minor._find_peak(self.i_residual)

            self.diag["major_cycles"].append({
                "peak_before"  : peak_before,
                "peak_after"   : peak_after,
                "cycle_thresh" : cycle_thresh,
                "global_thresh": global_thresh,
                "n_clean_iters": n_iters,
                "minor_trace"  : trace,
            })

            # ---- Write accumulated model back for next major cycle ----
            write_model_to_casa(
                f"{self.imagename}.model",
                self.i_model, self.q_model, self.u_model,
                template_path=f"{self.imagename}.residual",
            )
            logger.info(
                f"  Model → CASA: I_peak={np.max(np.abs(self.i_model)):.4f}"
            )

            if status == "converged":
                global_converged = True
                break
            if major_idx == self.max_major:
                logger.info("Reached max_major.")

        log_fh.close()

        # Final residual MFS snapshot
        self.diag["mfs_snapshots"].append({
            "I": np.nanmean(self.i_residual, axis=0).copy(),
            "Q": np.nanmean(self.q_residual, axis=0).copy(),
            "U": np.nanmean(self.u_residual, axis=0).copy(),
        })
        self.diag["params"] = self._param_dict()

        logger.info(
            f"Run finished ({'converged' if global_converged else 'max_major reached'})."
        )
        self._save(output_prefix)

        if debug:
            logger.info("--debug: generating diagnostic PDF …")
            from apclean.diagnostics import plot_all
            plot_all(self.diag, self.wcs_2d, self.freq_hz, output_prefix)

    # ------------------------------------------------------------------
    # Save FITS outputs
    # ------------------------------------------------------------------

    def _save(self, prefix: str) -> None:
        logger.info(f"Restoring and saving outputs → '{prefix}_*.fits' …")

        i_restored = restore_cube(self.i_model, self.i_residual, self.i_psf)
        q_restored = restore_cube(self.q_model, self.q_residual, self.i_psf)
        u_restored = restore_cube(self.u_model, self.u_residual, self.i_psf)

        # Channel cubes
        for arr, tag in [
            (i_restored,       "restored_I"),
            (q_restored,       "restored_Q"),
            (u_restored,       "restored_U"),
            (self.i_model,     "model_I"),
            (self.q_model,     "model_Q"),
            (self.u_model,     "model_U"),
            (self.i_residual,  "residual_I"),
            (self.q_residual,  "residual_Q"),
            (self.u_residual,  "residual_U"),
        ]:
            write_fits_cube(arr, self.wcs_2d, self.freq_hz,
                            f"{prefix}_{tag}.fits")

        # MFS 2-D images
        i_mfs, q_mfs, u_mfs = self._collapse_mfs(i_restored, q_restored, u_restored)
        p_mfs = np.sqrt(q_mfs ** 2 + u_mfs ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp_mfs = np.where(i_mfs > 0, p_mfs / i_mfs, np.nan)

        for arr, tag in [
            (i_mfs,  "mfs_I"),   (q_mfs,  "mfs_Q"),   (u_mfs,  "mfs_U"),
            (p_mfs,  "mfs_P"),   (fp_mfs, "mfs_frac_pol"),
        ]:
            write_fits_map(arr, self.wcs_2d, f"{prefix}_{tag}.fits")

        # VROOM parameter maps
        write_fits_map(self.rm_map,     self.wcs_2d, f"{prefix}_rm.fits")
        write_fits_map(self.p_amp_map,  self.wcs_2d, f"{prefix}_p.fits")
        write_fits_map(self.chi0_map,   self.wcs_2d, f"{prefix}_chi0.fits")
        write_fits_map(self.n_comp_map, self.wcs_2d, f"{prefix}_ncomp.fits")
        if np.any(np.isfinite(self.sigma_phi_map)):
            write_fits_map(self.sigma_phi_map, self.wcs_2d,
                           f"{prefix}_sigma_phi.fits")
        if np.any(np.isfinite(self.delta_phi_map)):
            write_fits_map(self.delta_phi_map, self.wcs_2d,
                           f"{prefix}_delta_phi.fits")

        logger.info(f"All outputs written.")
