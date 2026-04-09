"""
koala.cleaner
=============
Koala — main orchestrator.

Major cycle : casatasks.tclean(niter=0)
Minor cycle : VROOM-SBI spectral shape (all pixels) + Faraday (polarized)
Convergence : cycle-factor method  (cyclefactor × sidelobe × P0)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from koala.io import (
    read_casa_iquv,
    write_fits_cube,
    write_fits_map,
    write_model_to_casa,
)
from koala.minor_cycle import MinorCycle
from koala.psf import compute_sidelobe_level, restore_cube

logger = logging.getLogger(__name__)


class Koala:
    """
    Wideband Spectral + Polarimetric CLEAN.

    Parameters
    ----------
    ms               : str    Measurement set path.
    imagename        : str    tclean working image prefix.
    imsize           : int    Square image size in pixels.
    cell             : str    Cell size string, e.g. '15arcsec'.
    reffreq          : str    Reference frequency, e.g. '1.5GHz'.
    robust           : float  Briggs robust weighting (default 0.5).
    vroom_model_dir  : str    Path to trained VROOM-SBI models directory.
                              Must contain spectral_shape_posterior.pt.
                              For spectra+pol mode, also needs pol posteriors.
    mode             : str    'spectra' or 'spectra+pol' (default).
    loop_gain        : float  Minor-cycle loop gain (default 0.1).
    threshold_sigma  : float  Global stopping threshold in sigma_I (default 3).
    p_snr_threshold  : float  Polarization detection in sigma_P (default 5).
    cyclefactor      : float  Major-cycle trigger (default 1.5).
    sidelobe_radius  : int    Pixels around PSF peak excluded for sidelobe
                              estimation (default 15).
    niter_minor_max  : int    Hard cap on minor-cycle iterations (default 5000).
    max_major        : int    Maximum major cycles (default 10).
    n_samples        : int    VROOM-SBI posterior samples (default 1000).
    device           : str    VROOM device 'cuda' or 'cpu'.
    """

    VALID_MODES = ("spectra", "spectra+pol")

    def __init__(
        self,
        ms: str,
        imagename: str,
        imsize: int,
        cell: str,
        reffreq: str = "1.5GHz",
        robust: float = 0.5,
        vroom_model_dir: str = "models",
        mode: str = "spectra+pol",
        loop_gain: float = 0.1,
        threshold_sigma: float = 3.0,
        p_snr_threshold: float = 5.0,
        cyclefactor: float = 1.5,
        sidelobe_radius: int = 15,
        niter_minor_max: int = 5000,
        max_major: int = 10,
        n_samples: int = 1000,
        device: str = "cuda",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.ms = str(ms)
        self.imagename = str(imagename)
        self.imsize = imsize
        self.cell = cell
        self.reffreq = reffreq
        self.robust = robust
        self.mode = mode
        self.loop_gain = loop_gain
        self.threshold_sigma = threshold_sigma
        self.p_snr_threshold = p_snr_threshold
        self.cyclefactor = cyclefactor
        self.sidelobe_radius = sidelobe_radius
        self.niter_minor_max = niter_minor_max
        self.max_major = max_major
        self.n_samples = n_samples

        # Populated after first major cycle
        self.freq_hz: np.ndarray | None = None
        self.wcs_2d: WCS | None = None
        self.n_freq = self.n_dec = self.n_ra = 0
        self.sidelobe_level: float = 0.1

        # Working arrays (nfreq, ndec, nra)
        self.i_residual = self.q_residual = self.u_residual = None
        self.i_psf = None
        self.i_model = self.q_model = self.u_model = None
        self.sigma_i: np.ndarray | None = None
        self.mean_sigma_i: float = 0.0
        self.p_map: np.ndarray | None = None
        self.sigma_p: float = 0.0

        # VROOM parameter maps (ndec, nra) — polarized pixels
        self.rm_map: np.ndarray | None = None
        self.p_amp_map: np.ndarray | None = None
        self.chi0_map: np.ndarray | None = None
        self.n_comp_map: np.ndarray | None = None
        self.sigma_phi_map: np.ndarray | None = None
        self.delta_phi_map: np.ndarray | None = None

        # Spectral shape maps (ndec, nra) — all pixels
        self.log_F0_map: np.ndarray | None = None
        self.alpha_map: np.ndarray | None = None
        self.beta_map: np.ndarray | None = None
        self.gamma_map: np.ndarray | None = None

        # Diagnostics
        self.diag: dict = {
            "params": self._param_dict(),
            "sidelobe_level": None,
            "major_cycles": [],
            "mfs_snapshots": [],
            "source_pixel": None,
            "initial_dirty_i": None,
            "initial_dirty_q": None,
            "initial_dirty_u": None,
            "vroom_diag": {
                "q_norm": None, "u_norm": None, "result": None, "pixel": None,
            },
            "spectral_diag": {
                "samples": None, "i_model_spec": None, "pixel": None,
                "i_obs": None,
            },
        }

        # ── Load VROOM-SBI engine ──────────────────────────────────────
        logger.info("Loading VROOM-SBI engine …")
        from src.inference import InferenceEngine
        from src.simulator.physics import freq_to_lambda_sq

        self._freq_to_lambda_sq = freq_to_lambda_sq
        model_dir = Path(vroom_model_dir)

        self._engine = InferenceEngine(
            config=None, model_dir=str(model_dir), device=device
        )

        # Always load spectral shape model
        spec_path = model_dir / "spectral_shape_posterior.pt"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"Spectral shape model not found: {spec_path}"
            )
        self._engine.load_spectral_shape_model(spec_path)
        spec_meta = self._engine.posterior_metadata["spectral_shape"]
        self._train_freq_hz = np.array(spec_meta["freq_hz"])
        self._train_nu0 = float(spec_meta["nu0_hz"])
        logger.info(
            f"Spectral shape model: {len(self._train_freq_hz)} channels, "
            f"ν₀={self._train_nu0/1e6:.1f} MHz"
        )

        # Load pol models only for spectra+pol mode
        self._train_lambda_sq: np.ndarray | None = None
        if self.mode == "spectra+pol":
            self._engine.load_models()
            if self._engine.model_lambda_sq:
                first_key = next(iter(self._engine.model_lambda_sq))
                self._train_lambda_sq = np.asarray(
                    self._engine.model_lambda_sq[first_key]
                )

        self._need_freq_interp = False  # set after first _update_from_casa

        logger.info(f"Koala ready (mode={self.mode}).")

    def _param_dict(self) -> dict:
        return {
            k: getattr(self, k, None)
            for k in (
                "ms", "imagename", "imsize", "cell", "reffreq", "robust",
                "mode", "loop_gain", "threshold_sigma", "p_snr_threshold",
                "cyclefactor", "sidelobe_radius", "niter_minor_max", "max_major",
            )
        }

    # ------------------------------------------------------------------
    # Major cycle
    # ------------------------------------------------------------------

    def _run_major(self, first: bool) -> None:
        from casatasks import tclean

        stokes = "I" if self.mode == "spectra" else "IQUV"

        kw = dict(
            vis=self.ms,
            imagename=self.imagename,
            imsize=[self.imsize, self.imsize],
            cell=[self.cell],
            stokes=stokes,
            specmode="cube",
            nchan=-1,
            reffreq=self.reffreq,
            weighting="briggs",
            robust=self.robust,
            niter=0,
            deconvolver="hogbom",
            restoration=False,
        )
        if first:
            logger.info(
                f"Major cycle 1: computing PSF + dirty image (stokes={stokes}) …"
            )
            tclean(**kw, calcpsf=True, calcres=True)
        else:
            logger.info(
                f"Major cycle: re-gridding with updated model (stokes={stokes}) …"
            )
            tclean(**kw, calcpsf=False, calcres=True)

    # ------------------------------------------------------------------
    # Read residual and PSF from CASA images
    # ------------------------------------------------------------------

    def _update_from_casa(self) -> None:
        res_planes, freq_hz, wcs_2d = read_casa_iquv(
            f"{self.imagename}.residual"
        )

        if self.freq_hz is None:
            self.freq_hz = freq_hz
            self.wcs_2d = wcs_2d
            self.n_freq = len(freq_hz)
            self.n_dec = res_planes["I"].shape[1]
            self.n_ra = res_planes["I"].shape[2]
            logger.info(
                f"Grid: {self.n_freq} channels  {self.n_dec}×{self.n_ra} pixels"
            )

            shape3 = (self.n_freq, self.n_dec, self.n_ra)
            shape2 = (self.n_dec, self.n_ra)

            self.i_model = np.zeros(shape3)
            if self.mode == "spectra+pol":
                self.q_model = np.zeros(shape3)
                self.u_model = np.zeros(shape3)
                self.rm_map = np.full(shape2, np.nan)
                self.p_amp_map = np.full(shape2, np.nan)
                self.chi0_map = np.full(shape2, np.nan)
                self.n_comp_map = np.full(shape2, np.nan)
                self.sigma_phi_map = np.full(shape2, np.nan)
                self.delta_phi_map = np.full(shape2, np.nan)

            # Spectral shape maps — always
            self.log_F0_map = np.full(shape2, np.nan)
            self.alpha_map = np.full(shape2, np.nan)
            self.beta_map = np.full(shape2, np.nan)
            self.gamma_map = np.full(shape2, np.nan)

            psf_planes, _, _ = read_casa_iquv(f"{self.imagename}.psf")
            self.i_psf = psf_planes["I"]

            self.sidelobe_level = compute_sidelobe_level(
                self.i_psf, self.sidelobe_radius
            )
            self.diag["sidelobe_level"] = self.sidelobe_level

            # Frequency grids for VROOM
            cube_lsq = self._freq_to_lambda_sq(freq_hz)
            self._cube_lambda_sq = cube_lsq
            self._freq_sort_idx = self._match_frequencies(freq_hz)

            # Check if interpolation is needed
            if len(self._train_freq_hz) != len(freq_hz):
                self._need_freq_interp = True
                logger.info(
                    f"Frequency grids differ in length "
                    f"(obs={len(freq_hz)}, train={len(self._train_freq_hz)}) "
                    f"— interpolation enabled"
                )
            elif not np.allclose(
                np.sort(freq_hz), np.sort(self._train_freq_hz), rtol=1e-3
            ):
                self._need_freq_interp = True
                logger.info("Frequency grids differ — interpolation enabled")
            else:
                self._need_freq_interp = False

        self.i_residual = res_planes["I"]
        if self.mode == "spectra+pol":
            self.q_residual = res_planes.get(
                "Q", np.zeros_like(self.i_residual)
            )
            self.u_residual = res_planes.get(
                "U", np.zeros_like(self.i_residual)
            )
        else:
            self.q_residual = None
            self.u_residual = None

        logger.info(
            f"  Residual I peak = "
            f"{np.nanmax(np.abs(self.i_residual)):.4f} Jy/beam"
        )

    def _match_frequencies(self, freq_hz: np.ndarray) -> np.ndarray | None:
        """Check if cube frequencies are reversed w.r.t. VROOM pol model."""
        if not self._engine.model_lambda_sq:
            return None
        from src.simulator.physics import freq_to_lambda_sq

        ref_lsq = next(iter(self._engine.model_lambda_sq.values()))
        cube_lsq = freq_to_lambda_sq(freq_hz)
        n = len(cube_lsq)
        if len(ref_lsq) != n:
            logger.warning(
                "Channel count mismatch between cube and VROOM pol model."
            )
            return None
        if np.allclose(cube_lsq, ref_lsq, rtol=1e-2):
            return None
        if np.allclose(cube_lsq[::-1], ref_lsq, rtol=1e-2):
            logger.info("Frequencies reversed — reordering for VROOM.")
            return np.arange(n)[::-1]
        logger.warning(
            "Frequency grid mismatch — VROOM RM values may be unreliable."
        )
        return None

    # ------------------------------------------------------------------
    # Noise estimation
    # ------------------------------------------------------------------

    def _estimate_noise(self) -> None:
        sigma_i = np.zeros(self.n_freq)
        for c in range(self.n_freq):
            ch = self.i_residual[c]
            med = np.nanmedian(ch)
            sigma_i[c] = 1.4826 * np.nanmedian(np.abs(ch - med))
        self.sigma_i = sigma_i
        self.mean_sigma_i = float(np.nanmean(sigma_i))

        if self.mode == "spectra+pol" and self.q_residual is not None:
            p_map = np.nanmean(
                np.sqrt(self.q_residual ** 2 + self.u_residual ** 2), axis=0
            )
            self.p_map = p_map
            p_med = np.nanmedian(p_map)
            self.sigma_p = 1.4826 * float(
                np.nanmedian(np.abs(p_map - p_med))
            )
            logger.info(
                f"  sigma_I={self.mean_sigma_i:.4f}  sigma_P={self.sigma_p:.4f}  "
                f"P_thresh({self.p_snr_threshold}σ)="
                f"{self.p_snr_threshold * self.sigma_p:.4f}"
            )
        else:
            self.p_map = None
            self.sigma_p = 0.0
            logger.info(f"  sigma_I={self.mean_sigma_i:.4f}")

    # ------------------------------------------------------------------
    # Store parameter maps from minor cycle
    # ------------------------------------------------------------------

    def _store_vroom_maps(self, vroom_diag: dict) -> None:
        result = vroom_diag.get("result")
        pixel = vroom_diag.get("pixel")
        if result is None or pixel is None:
            return
        y, x = pixel
        self.n_comp_map[y, x] = result.n_components
        if result.components:
            comp = result.components[0]
            self.rm_map[y, x] = comp.rm_mean
            self.p_amp_map[y, x] = np.sqrt(comp.q_mean ** 2 + comp.u_mean ** 2)
            if comp.chi0_mean is not None:
                self.chi0_map[y, x] = comp.chi0_mean
            if comp.sigma_phi_mean is not None:
                self.sigma_phi_map[y, x] = comp.sigma_phi_mean
            if comp.delta_phi_mean is not None:
                self.delta_phi_map[y, x] = comp.delta_phi_mean

    def _store_spectral_maps(self, spectral_diag: dict) -> None:
        samples = spectral_diag.get("samples")
        pixel = spectral_diag.get("pixel")
        if samples is None or pixel is None:
            return
        y, x = pixel
        theta_mean = np.mean(samples, axis=0)
        self.log_F0_map[y, x] = theta_mean[0]
        self.alpha_map[y, x] = theta_mean[1]
        self.beta_map[y, x] = theta_mean[2]
        self.gamma_map[y, x] = theta_mean[3]

    # ------------------------------------------------------------------
    # MFS collapse (σ_I-weighted channel mean)
    # ------------------------------------------------------------------

    def _collapse_mfs(
        self,
        i_cube: np.ndarray,
        q_cube: np.ndarray | None = None,
        u_cube: np.ndarray | None = None,
    ) -> tuple:
        valid = self.sigma_i > 0
        w = np.where(valid, 1.0 / self.sigma_i ** 2, 0.0)
        wsum = w.sum()
        i_mfs = np.einsum("c,cyx->yx", w, i_cube) / wsum
        if q_cube is not None and u_cube is not None:
            q_mfs = np.einsum("c,cyx->yx", w, q_cube) / wsum
            u_mfs = np.einsum("c,cyx->yx", w, u_cube) / wsum
            return i_mfs, q_mfs, u_mfs
        return (i_mfs,)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, output_prefix: str = "koala", debug: bool = False) -> None:
        log_path = f"{output_prefix}.log"
        log_fh = open(log_path, "w")
        log_fh.write(
            f"KOALA  MS={self.ms}  mode={self.mode}\n"
            f"cyclefactor={self.cyclefactor}  sidelobe_radius={self.sidelobe_radius}\n"
            f"loop_gain={self.loop_gain}  threshold={self.threshold_sigma}σ  "
            f"p_threshold={self.p_snr_threshold}σ\n"
            f"{'iter':<8}{'y':<5}{'x':<5}{'peak':<14}{'pol':<5}model\n"
            + "-" * 70 + "\n"
        )

        # MinorCycle is created after first major cycle populates freq info
        minor: MinorCycle | None = None

        global_converged = False

        for major_idx in range(1, self.max_major + 1):
            logger.info(f"=== Major cycle {major_idx}/{self.max_major} ===")
            log_fh.write(f"\n--- Major cycle {major_idx} ---\n")

            self._run_major(first=(major_idx == 1))
            self._update_from_casa()
            self._estimate_noise()

            # Create MinorCycle after first update (freq info now available)
            if minor is None:
                minor = MinorCycle(
                    engine=self._engine,
                    freq_hz=self.freq_hz,
                    cube_lambda_sq=self._cube_lambda_sq,
                    freq_sort_idx=self._freq_sort_idx,
                    train_freq_hz=self._train_freq_hz,
                    train_nu0=self._train_nu0,
                    train_lambda_sq=self._train_lambda_sq,
                    need_freq_interp=self._need_freq_interp,
                    mode=self.mode,
                    loop_gain=self.loop_gain,
                    p_snr_threshold=self.p_snr_threshold,
                    n_samples=self.n_samples,
                )

            # MFS snapshot
            snap = {"I": np.nanmean(self.i_residual, axis=0).copy()}
            if self.mode == "spectra+pol":
                snap["Q"] = np.nanmean(self.q_residual, axis=0).copy()
                snap["U"] = np.nanmean(self.u_residual, axis=0).copy()
            self.diag["mfs_snapshots"].append(snap)

            # Save initial dirty spectra at source pixel (once)
            if major_idx == 1:
                m = np.nanmean(np.abs(self.i_residual), axis=0)
                y_src, x_src = np.unravel_index(np.nanargmax(m), m.shape)
                self.diag["source_pixel"] = (int(y_src), int(x_src))
                self.diag["initial_dirty_i"] = (
                    self.i_residual[:, y_src, x_src].copy()
                )
                if self.mode == "spectra+pol":
                    self.diag["initial_dirty_q"] = (
                        self.q_residual[:, y_src, x_src].copy()
                    )
                    self.diag["initial_dirty_u"] = (
                        self.u_residual[:, y_src, x_src].copy()
                    )

            global_thresh = self.threshold_sigma * self.mean_sigma_i
            peak_before = float(np.nanmax(np.abs(self.i_residual)))

            if major_idx > 1 and peak_before < global_thresh:
                logger.info("Peak already below threshold — done.")
                global_converged = True
                break

            p0 = peak_before
            cycle_thresh = self.cyclefactor * self.sidelobe_level * p0
            logger.info(
                f"  P0={p0:.4f}  cycle_thresh={cycle_thresh:.4f}  "
                f"global_thresh={global_thresh:.4f}"
            )

            # ── Minor cycle ────────────────────────────────────────────
            vroom_diag_this = {
                "q_norm": None, "u_norm": None, "result": None, "pixel": None,
            }
            spectral_diag_this = {
                "samples": None, "i_model_spec": None, "pixel": None,
                "i_obs": None,
            }

            n_iters, trace, status = minor.run(
                i_residual=self.i_residual,
                q_residual=self.q_residual,
                u_residual=self.u_residual,
                i_model=self.i_model,
                q_model=self.q_model,
                u_model=self.u_model,
                i_psf=self.i_psf,
                p_map=self.p_map,
                sigma_p=self.sigma_p,
                cycle_thresh=cycle_thresh,
                global_thresh=global_thresh,
                niter_max=self.niter_minor_max,
                log_fh=log_fh,
                vroom_diag=vroom_diag_this,
                spectral_diag=spectral_diag_this,
            )

            # Store maps and diagnostics
            self._store_spectral_maps(spectral_diag_this)
            if self.mode == "spectra+pol":
                self._store_vroom_maps(vroom_diag_this)
            if (self.diag["vroom_diag"]["q_norm"] is None
                    and vroom_diag_this["q_norm"] is not None):
                self.diag["vroom_diag"] = vroom_diag_this
            if (self.diag["spectral_diag"]["samples"] is None
                    and spectral_diag_this["samples"] is not None):
                self.diag["spectral_diag"] = spectral_diag_this

            _, _, peak_after = minor._find_peak(self.i_residual)

            self.diag["major_cycles"].append({
                "peak_before": peak_before,
                "peak_after": peak_after,
                "cycle_thresh": cycle_thresh,
                "global_thresh": global_thresh,
                "n_clean_iters": n_iters,
                "minor_trace": trace,
            })

            # Write accumulated model back for next major cycle
            write_model_to_casa(
                f"{self.imagename}.model",
                self.i_model,
                self.q_model if self.mode == "spectra+pol" else np.zeros_like(self.i_model),
                self.u_model if self.mode == "spectra+pol" else np.zeros_like(self.i_model),
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
        snap = {"I": np.nanmean(self.i_residual, axis=0).copy()}
        if self.mode == "spectra+pol" and self.q_residual is not None:
            snap["Q"] = np.nanmean(self.q_residual, axis=0).copy()
            snap["U"] = np.nanmean(self.u_residual, axis=0).copy()
        self.diag["mfs_snapshots"].append(snap)
        self.diag["params"] = self._param_dict()

        logger.info(
            f"Run finished "
            f"({'converged' if global_converged else 'max_major reached'})."
        )
        self._save(output_prefix)

        if debug:
            logger.info("--debug: generating diagnostic PDF …")
            from koala.diagnostics import plot_all
            plot_all(self.diag, self.wcs_2d, self.freq_hz, output_prefix)

    # ------------------------------------------------------------------
    # Save FITS outputs
    # ------------------------------------------------------------------

    def _save(self, prefix: str) -> None:
        logger.info(f"Restoring and saving outputs → '{prefix}_*.fits' …")

        i_restored = restore_cube(self.i_model, self.i_residual, self.i_psf)

        # ── Channel cubes ──────────────────────────────────────────────
        for arr, tag in [
            (i_restored, "restored_I"),
            (self.i_model, "model_I"),
            (self.i_residual, "residual_I"),
        ]:
            write_fits_cube(arr, self.wcs_2d, self.freq_hz, f"{prefix}_{tag}.fits")

        if self.mode == "spectra+pol" and self.q_model is not None:
            q_restored = restore_cube(self.q_model, self.q_residual, self.i_psf)
            u_restored = restore_cube(self.u_model, self.u_residual, self.i_psf)
            for arr, tag in [
                (q_restored, "restored_Q"),
                (u_restored, "restored_U"),
                (self.q_model, "model_Q"),
                (self.u_model, "model_U"),
                (self.q_residual, "residual_Q"),
                (self.u_residual, "residual_U"),
            ]:
                write_fits_cube(
                    arr, self.wcs_2d, self.freq_hz, f"{prefix}_{tag}.fits"
                )

        # ── MFS 2-D images ────────────────────────────────────────────
        if self.mode == "spectra+pol" and self.q_model is not None:
            i_mfs, q_mfs, u_mfs = self._collapse_mfs(
                i_restored, q_restored, u_restored
            )
            p_mfs = np.sqrt(q_mfs ** 2 + u_mfs ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                fp_mfs = np.where(i_mfs > 0, p_mfs / i_mfs, np.nan)

            for arr, tag in [
                (i_mfs, "mfs_I"), (q_mfs, "mfs_Q"), (u_mfs, "mfs_U"),
                (p_mfs, "mfs_P"), (fp_mfs, "mfs_frac_pol"),
            ]:
                write_fits_map(arr, self.wcs_2d, f"{prefix}_{tag}.fits")
        else:
            (i_mfs,) = self._collapse_mfs(i_restored)
            write_fits_map(i_mfs, self.wcs_2d, f"{prefix}_mfs_I.fits")

        # ── Spectral shape parameter maps (both modes) ────────────────
        write_fits_map(self.log_F0_map, self.wcs_2d, f"{prefix}_log_F0.fits")
        write_fits_map(self.alpha_map, self.wcs_2d, f"{prefix}_alpha.fits")
        write_fits_map(self.beta_map, self.wcs_2d, f"{prefix}_beta.fits")
        write_fits_map(self.gamma_map, self.wcs_2d, f"{prefix}_gamma.fits")

        # ── VROOM polarization maps (spectra+pol only) ────────────────
        if self.mode == "spectra+pol" and self.rm_map is not None:
            write_fits_map(self.rm_map, self.wcs_2d, f"{prefix}_rm.fits")
            write_fits_map(self.p_amp_map, self.wcs_2d, f"{prefix}_p.fits")
            write_fits_map(self.chi0_map, self.wcs_2d, f"{prefix}_chi0.fits")
            write_fits_map(
                self.n_comp_map, self.wcs_2d, f"{prefix}_ncomp.fits"
            )
            if np.any(np.isfinite(self.sigma_phi_map)):
                write_fits_map(
                    self.sigma_phi_map, self.wcs_2d,
                    f"{prefix}_sigma_phi.fits",
                )
            if np.any(np.isfinite(self.delta_phi_map)):
                write_fits_map(
                    self.delta_phi_map, self.wcs_2d,
                    f"{prefix}_delta_phi.fits",
                )

        logger.info("All outputs written.")
