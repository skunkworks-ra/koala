"""
koala.minor_cycle
=================
MinorCycle — per-pixel CLEAN with spectral shape inference.

Every pixel gets a VROOM-SBI spectral shape fit for I(ν).
In spectra+pol mode, polarized pixels additionally get Faraday Q/U inference
using the spectral I model for normalization.

The class is stateless between major cycles: residuals and model arrays are
passed in and modified in-place.
"""

from __future__ import annotations

import logging

import numpy as np

from koala.psf import subtract_psf_at

logger = logging.getLogger(__name__)


def _interp_nan_safe(
    obs_spectrum: np.ndarray,
    obs_grid: np.ndarray,
    train_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate an observed spectrum onto a training frequency grid,
    handling NaN channels.

    NaN/non-finite channels are excluded from interpolation.  Training
    channels that fall outside the observed range are zeroed with weight=0
    (matching vroom-sbi flagged-channel convention).

    Parameters
    ----------
    obs_spectrum : (n_obs,)  — may contain NaNs
    obs_grid     : (n_obs,)  — observation grid (Hz or m²), must be ascending
    train_grid   : (n_train,) — target grid, must be ascending

    Returns
    -------
    interp_spectrum : (n_train,)
    weights         : (n_train,)  — 1.0 good, 0.0 flagged
    """
    good = np.isfinite(obs_spectrum) & np.isfinite(obs_grid)
    if not np.any(good):
        return np.zeros(len(train_grid)), np.zeros(len(train_grid))

    obs_g = obs_grid[good]
    obs_s = obs_spectrum[good]

    interp_vals = np.interp(train_grid, obs_g, obs_s)

    weights = np.ones(len(train_grid))
    obs_min, obs_max = obs_g.min(), obs_g.max()
    outside = (train_grid < obs_min) | (train_grid > obs_max)
    weights[outside] = 0.0
    interp_vals[outside] = 0.0

    return interp_vals, weights


class MinorCycle:
    """
    Parameters
    ----------
    engine           : InferenceEngine  — loaded VROOM-SBI engine (has both
                       spectral shape and pol posteriors)
    freq_hz          : (nfreq,) float64 — observation channel frequencies
    cube_lambda_sq   : (nfreq,) float64 — λ² for each channel
    freq_sort_idx    : ndarray or None  — reorder index if freq grid is reversed
    train_freq_hz    : (n_train,) float64 — training frequency grid
    train_nu0        : float — reference frequency from spectral shape model
    train_lambda_sq  : (n_train_pol,) or None — training λ² grid for pol model
    need_freq_interp : bool — True if obs and train freq grids differ
    mode             : str  — "spectra" or "spectra+pol"
    loop_gain        : float
    p_snr_threshold  : float — P detection threshold in sigma_P
    n_samples        : int   — VROOM posterior samples
    """

    def __init__(
        self,
        engine,
        freq_hz: np.ndarray,
        cube_lambda_sq: np.ndarray,
        freq_sort_idx: np.ndarray | None,
        train_freq_hz: np.ndarray,
        train_nu0: float,
        train_lambda_sq: np.ndarray | None,
        need_freq_interp: bool,
        mode: str = "spectra+pol",
        loop_gain: float = 0.1,
        p_snr_threshold: float = 5.0,
        n_samples: int = 1000,
    ):
        self.engine = engine
        self.freq_hz = freq_hz
        self.cube_lambda_sq = cube_lambda_sq
        self.freq_sort_idx = freq_sort_idx
        self.train_freq_hz = train_freq_hz
        self.train_nu0 = train_nu0
        self.train_lambda_sq = train_lambda_sq
        self.need_freq_interp = need_freq_interp
        self.mode = mode
        self.loop_gain = loop_gain
        self.p_snr_threshold = p_snr_threshold
        self.n_samples = n_samples
        self.n_freq = len(freq_hz)

        # Precompute log(ν/ν₀) at observation frequencies for analytic SED eval
        self._log_nu_ratio_obs = np.log(self.freq_hz / self.train_nu0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        i_residual: np.ndarray,       # (nfreq, ndec, nra) — modified in-place
        q_residual: np.ndarray | None,
        u_residual: np.ndarray | None,
        i_model: np.ndarray,           # (nfreq, ndec, nra) — accumulated in-place
        q_model: np.ndarray | None,
        u_model: np.ndarray | None,
        i_psf: np.ndarray,             # (nfreq, ndec, nra)
        p_map: np.ndarray | None,      # (ndec, nra) — updated in-place
        sigma_p: float,
        cycle_thresh: float,
        global_thresh: float,
        niter_max: int = 5000,
        log_fh=None,
        vroom_diag: dict | None = None,
        spectral_diag: dict | None = None,
    ) -> tuple[int, list, str]:
        """
        Run one minor-cycle pass.

        Returns
        -------
        n_iters : int
        trace   : list of (iter, peak, y, x)
        status  : str  — 'converged', 'need_major', or 'cap'
        """
        trace = []
        n_pol = n_spectra = 0
        status = "cap"

        for it in range(niter_max):
            y, x, peak = self._find_peak(i_residual)
            trace.append((it, peak, y, x))

            if peak < global_thresh:
                status = "converged"
                logger.info(
                    f"    iter {it}: peak={peak:.4f} < global_thresh → CONVERGED"
                )
                break
            if peak < cycle_thresh:
                status = "need_major"
                logger.info(
                    f"    iter {it}: peak={peak:.4f} < cycle_thresh → major cycle"
                )
                break

            # ── Spectral shape inference (always) ──────────────────────
            try:
                spec_samples, i_model_spec = self._run_spectral_shape(
                    i_residual, y, x
                )
            except Exception as exc:
                logger.warning(
                    f"    Spectral shape failed at ({y},{x}): {exc} — flat fallback"
                )
                spec_samples = None
                i_model_spec = np.full(
                    self.n_freq,
                    np.nanmean(np.abs(i_residual[:, y, x])),
                )

            # Store first spectral result for diagnostics
            if (spectral_diag is not None
                    and spectral_diag.get("samples") is None
                    and spec_samples is not None):
                spectral_diag["samples"] = spec_samples.copy()
                spectral_diag["i_model_spec"] = i_model_spec.copy()
                spectral_diag["pixel"] = (y, x)
                spectral_diag["i_obs"] = i_residual[:, y, x].copy()

            # ── Check polarization (spectra+pol mode only) ─────────────
            polarized = False
            if (self.mode == "spectra+pol"
                    and p_map is not None
                    and q_residual is not None
                    and u_residual is not None):
                polarized = p_map[y, x] >= self.p_snr_threshold * sigma_p

            if polarized:
                try:
                    result, q_norm, u_norm = self._run_vroom(
                        i_model_spec, q_residual, u_residual, y, x
                    )
                except Exception as exc:
                    logger.warning(
                        f"    VROOM pol failed at ({y},{x}): {exc} — spectra-only"
                    )
                    polarized = False

            # ── PSF subtraction ────────────────────────────────────────
            # Frequency-dependent amplitude from spectral shape model:
            # scale the SED shape so its mean equals loop_gain × peak
            mean_model = np.mean(np.abs(i_model_spec))
            if mean_model > 0:
                scale = self.loop_gain * float(
                    np.nanmean(np.abs(i_residual[:, y, x]))
                )
                i_amp = (scale / mean_model) * i_model_spec
            else:
                i_amp = np.full(
                    self.n_freq,
                    self.loop_gain * float(
                        np.nanmean(np.abs(i_residual[:, y, x]))
                    ),
                )

            if polarized:
                for c in range(self.n_freq):
                    subtract_psf_at(i_residual[c], i_psf[c], y, x, i_amp[c])
                    subtract_psf_at(
                        q_residual[c], i_psf[c], y, x, q_norm[c] * i_amp[c]
                    )
                    subtract_psf_at(
                        u_residual[c], i_psf[c], y, x, u_norm[c] * i_amp[c]
                    )
                i_model[:, y, x] += i_amp
                q_model[:, y, x] += q_norm * i_amp
                u_model[:, y, x] += u_norm * i_amp
                n_pol += 1
                model_tag = (
                    f"{result.model_type} nc={result.n_components} "
                    f"RM={result.components[0].rm_mean:.1f}"
                )

                # Store first VROOM result for diagnostics
                if vroom_diag is not None and vroom_diag.get("q_norm") is None:
                    vroom_diag["q_norm"] = q_norm.copy()
                    vroom_diag["u_norm"] = u_norm.copy()
                    vroom_diag["result"] = result
                    vroom_diag["pixel"] = (y, x)

            else:
                for c in range(self.n_freq):
                    subtract_psf_at(i_residual[c], i_psf[c], y, x, i_amp[c])
                i_model[:, y, x] += i_amp
                n_spectra += 1
                if spec_samples is not None:
                    theta = np.mean(spec_samples, axis=0)
                    model_tag = f"SED α={theta[1]:.2f} β={theta[2]:.2f}"
                else:
                    model_tag = "flat-fallback"

            # Update P map at cleaned pixel (if doing pol)
            if (p_map is not None
                    and q_residual is not None
                    and u_residual is not None):
                p_map[y, x] = float(np.nanmean(np.sqrt(
                    q_residual[:, y, x] ** 2 + u_residual[:, y, x] ** 2
                )))

            if log_fh:
                log_fh.write(
                    f"{it:<8d}{y:<5d}{x:<5d}{peak:<14.6f}"
                    f"{'P' if polarized else 'S':<5}{model_tag}\n"
                )
                log_fh.flush()

            if it % 100 == 0 or it < 5:
                logger.info(
                    f"    iter {it:5d}: peak={peak:.4f} at ({y},{x}) "
                    f"[{model_tag}]"
                )

        logger.info(
            f"  Minor cycle: {n_pol} pol + {n_spectra} spectra-only "
            f"= {n_pol + n_spectra} iterations  status={status}"
        )
        return n_pol + n_spectra, trace, status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peak(
        i_residual: np.ndarray,
    ) -> tuple[int, int, float]:
        m = np.nanmean(np.abs(i_residual), axis=0)
        idx = np.unravel_index(np.nanargmax(m), m.shape)
        y, x = int(idx[0]), int(idx[1])
        return y, x, float(m[y, x])

    def _run_spectral_shape(
        self,
        i_residual: np.ndarray,
        y: int,
        x: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        VROOM-SBI spectral shape inference at pixel (y, x).

        Returns
        -------
        samples      : (n_samples, 4) — [log_F0, alpha, beta, gamma]
        i_model_spec : (n_freq,) — noiseless I(ν) at observation frequencies
        """
        i_spec = i_residual[:, y, x].copy()

        if self.need_freq_interp:
            # Sort obs grid ascending for np.interp
            sort_asc = np.argsort(self.freq_hz)
            i_spec_infer, weights = _interp_nan_safe(
                i_spec[sort_asc], self.freq_hz[sort_asc], self.train_freq_hz
            )
            samples = self.engine.infer_spectra(
                i_spec_infer, weights=weights, n_samples=self.n_samples
            )
        else:
            # Same grid — zero out NaN channels (vroom-sbi convention)
            i_spec_infer = i_spec.copy()
            i_spec_infer[~np.isfinite(i_spec_infer)] = 0.0
            samples = self.engine.infer_spectra(
                i_spec_infer, n_samples=self.n_samples
            )

        # Posterior mean → analytic model at observation frequencies
        theta_mean = np.mean(samples, axis=0)
        x_arr = self._log_nu_ratio_obs
        log_F = (
            theta_mean[0]
            + theta_mean[1] * x_arr
            + theta_mean[2] * x_arr ** 2
            + theta_mean[3] * x_arr ** 3
        )
        i_model_spec = np.exp(log_F)

        return samples, i_model_spec

    def _run_vroom(
        self,
        i_model_spec: np.ndarray,
        q_residual: np.ndarray,
        u_residual: np.ndarray,
        y: int,
        x: int,
    ):
        """
        VROOM-SBI Faraday rotation inference at pixel (y, x).

        Uses i_model_spec (from spectral shape) for Q/I, U/I normalization
        instead of raw residual I.

        Returns (result, q_norm, u_norm) in cube channel order.
        """
        from src.simulator.base_simulator import RMSimulator

        q_spec = q_residual[:, y, x].copy()
        u_spec = u_residual[:, y, x].copy()

        # Normalize Q/U by spectral model I (clean, not noisy residual)
        valid = (i_model_spec > 0) & np.isfinite(i_model_spec)
        q_over_i = np.nan_to_num(np.where(valid, q_spec / i_model_spec, 0.0))
        u_over_i = np.nan_to_num(np.where(valid, u_spec / i_model_spec, 0.0))

        # Reorder to training grid if cube is reversed
        if self.freq_sort_idx is not None:
            q_norm_infer = q_over_i[self.freq_sort_idx]
            u_norm_infer = u_over_i[self.freq_sort_idx]
        else:
            q_norm_infer = q_over_i.copy()
            u_norm_infer = u_over_i.copy()

        # Interpolate Q/I, U/I onto training λ² grid if freq grids differ
        if self.need_freq_interp and self.train_lambda_sq is not None:
            obs_lsq = self.cube_lambda_sq.copy()
            if self.freq_sort_idx is not None:
                obs_lsq = obs_lsq[self.freq_sort_idx]

            # Both grids must be ascending for interp
            sort_obs = np.argsort(obs_lsq)
            sort_train = np.argsort(self.train_lambda_sq)
            train_lsq_sorted = self.train_lambda_sq[sort_train]

            q_interp, _ = _interp_nan_safe(
                q_norm_infer[sort_obs], obs_lsq[sort_obs], train_lsq_sorted
            )
            u_interp, _ = _interp_nan_safe(
                u_norm_infer[sort_obs], obs_lsq[sort_obs], train_lsq_sorted
            )

            # Restore training grid order
            unsort_train = np.argsort(sort_train)
            q_norm_infer = q_interp[unsort_train]
            u_norm_infer = u_interp[unsort_train]

        # Zero NaNs (vroom-sbi convention)
        q_norm_infer = np.nan_to_num(q_norm_infer)
        u_norm_infer = np.nan_to_num(u_norm_infer)

        # VROOM inference: posterior p(θ | Q/I, U/I)
        qu_obs = np.concatenate([q_norm_infer, u_norm_infer])
        best_result, _ = self.engine.infer(qu_obs, n_samples=self.n_samples)

        # Posterior mean → noiseless model spectra at OBSERVATION frequencies
        theta_mean = np.mean(best_result.all_samples, axis=0)

        sim = RMSimulator.__new__(RMSimulator)
        sim.freq = self.freq_hz.copy()
        sim.lambda_sq = self.cube_lambda_sq.copy()
        sim._n_freq = self.n_freq
        sim._weights = np.ones(self.n_freq)
        sim.n_components = best_result.n_components
        sim.model_type = best_result.model_type
        sim._params_per_comp = 3 if sim.model_type == "faraday_thin" else 4
        sim._n_params = sim._params_per_comp * sim.n_components

        qu_model = sim.simulate_noiseless(theta_mean)
        q_mod = qu_model[: self.n_freq]
        u_mod = qu_model[self.n_freq:]

        return best_result, q_mod, u_mod
