"""
apclean.minor_cycle
===================
MinorCycle — per-pixel CLEAN with two paths:

  Polarized   (P > p_snr × sigma_P) : VROOM-SBI → frequency-dependent Q/U model
  Unpolarized (I only)               : Hogbom flat-spectrum subtraction

The class is stateless between major cycles: residuals and model arrays are
passed in and modified in-place.  Diagnostics are returned as a dict.
"""

from __future__ import annotations

import logging

import numpy as np

from apclean.psf import subtract_psf_at

logger = logging.getLogger(__name__)


class MinorCycle:
    """
    Parameters
    ----------
    engine           : InferenceEngine  — loaded VROOM-SBI engine
    freq_hz          : (nfreq,) float64 — channel frequencies
    cube_lambda_sq   : (nfreq,) float64 — λ² for each channel
    freq_sort_idx    : ndarray or None  — reorder index if freq grid is reversed
    loop_gain        : float
    p_snr_threshold  : float            — P detection threshold in sigma_P
    n_samples        : int              — VROOM posterior samples
    """

    def __init__(
        self,
        engine          ,
        freq_hz         : np.ndarray,
        cube_lambda_sq  : np.ndarray,
        freq_sort_idx   : np.ndarray | None,
        loop_gain       : float = 0.1,
        p_snr_threshold : float = 5.0,
        n_samples       : int   = 1000,
    ):
        self.engine          = engine
        self.freq_hz         = freq_hz
        self.cube_lambda_sq  = cube_lambda_sq
        self.freq_sort_idx   = freq_sort_idx
        self.loop_gain       = loop_gain
        self.p_snr_threshold = p_snr_threshold
        self.n_samples       = n_samples
        self.n_freq          = len(freq_hz)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        i_residual    : np.ndarray,   # (nfreq, ndec, nra)  — modified in-place
        q_residual    : np.ndarray,
        u_residual    : np.ndarray,
        i_model       : np.ndarray,   # (nfreq, ndec, nra)  — accumulated in-place
        q_model       : np.ndarray,
        u_model       : np.ndarray,
        i_psf         : np.ndarray,   # (nfreq, ndec, nra)
        p_map         : np.ndarray,   # (ndec, nra)  — updated in-place
        sigma_p       : float,
        cycle_thresh  : float,        # → trigger major cycle
        global_thresh : float,        # → fully converged
        niter_max     : int   = 5000,
        log_fh        = None,
        vroom_diag    : dict  | None = None,   # filled with source-pixel VROOM info
    ) -> tuple[int, list, str]:
        """
        Run one minor-cycle pass.

        Stops when:
          peak < cycle_thresh  → status='need_major'
          peak < global_thresh → status='converged'
          niter_max reached    → status='cap'

        Returns
        -------
        n_iters : int
        trace   : list of (iter, peak, y, x)
        status  : str
        """
        # noise for P map from residuals
        q2u2 = np.sqrt(q_residual ** 2 + u_residual ** 2)
        # keep p_map view consistent — it's passed in from cleaner

        trace  = []
        n_pol  = n_unpol = 0
        status = "cap"

        for it in range(niter_max):
            y, x, peak = self._find_peak(i_residual)
            trace.append((it, peak, y, x))

            if peak < global_thresh:
                status = "converged"
                logger.info(
                    f"    iter {it}: peak={peak:.4f} < global_thresh "
                    f"→ CONVERGED"
                )
                break
            if peak < cycle_thresh:
                status = "need_major"
                logger.info(
                    f"    iter {it}: peak={peak:.4f} < cycle_thresh "
                    f"→ major cycle"
                )
                break

            polarized = p_map[y, x] >= self.p_snr_threshold * sigma_p

            if polarized:
                try:
                    result, q_norm, u_norm = self._run_vroom(
                        i_residual, q_residual, u_residual, y, x
                    )
                except Exception as exc:
                    logger.warning(
                        f"    VROOM failed at ({y},{x}): {exc} — I-only fallback"
                    )
                    polarized = False

            if polarized:
                amp = self.loop_gain * float(np.mean(i_residual[:, y, x]))
                for c in range(self.n_freq):
                    subtract_psf_at(i_residual[c], i_psf[c], y, x, amp)
                    subtract_psf_at(q_residual[c], i_psf[c], y, x, q_norm[c] * amp)
                    subtract_psf_at(u_residual[c], i_psf[c], y, x, u_norm[c] * amp)
                i_model[:, y, x] += amp
                q_model[:, y, x] += q_norm * amp
                u_model[:, y, x] += u_norm * amp
                n_pol += 1
                model_tag = (
                    f"{result.model_type} nc={result.n_components} "
                    f"RM={result.components[0].rm_mean:.1f}"
                )

                # Store first VROOM result at the brightest source pixel
                if vroom_diag is not None and vroom_diag.get("q_norm") is None:
                    vroom_diag["q_norm"]  = q_norm.copy()
                    vroom_diag["u_norm"]  = u_norm.copy()
                    vroom_diag["result"]  = result
                    vroom_diag["pixel"]   = (y, x)

            else:
                amp = self.loop_gain * float(np.mean(i_residual[:, y, x]))
                for c in range(self.n_freq):
                    subtract_psf_at(i_residual[c], i_psf[c], y, x, amp)
                i_model[:, y, x] += amp
                n_unpol += 1
                model_tag = "I-only"

            # Update P map at cleaned pixel
            p_map[y, x] = float(
                np.nanmean(
                    np.sqrt(q_residual[:, y, x] ** 2 + u_residual[:, y, x] ** 2)
                )
            )

            if log_fh:
                log_fh.write(
                    f"{it:<8d}{y:<5d}{x:<5d}{peak:<14.6f}"
                    f"{'Y' if polarized else 'N':<5}{model_tag}\n"
                )
                log_fh.flush()

            if it % 100 == 0 or it < 5:
                logger.info(
                    f"    iter {it:5d}: peak={peak:.4f} at ({y},{x}) "
                    f"[{model_tag}]"
                )

        logger.info(
            f"  Minor cycle: {n_pol} pol + {n_unpol} I-only "
            f"= {n_pol + n_unpol} CLEAN iterations  status={status}"
        )
        return n_pol + n_unpol, trace, status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peak(
        i_residual: np.ndarray,
    ) -> tuple[int, int, float]:
        m   = np.nanmean(np.abs(i_residual), axis=0)
        idx = np.unravel_index(np.nanargmax(m), m.shape)
        y, x = int(idx[0]), int(idx[1])
        return y, x, float(m[y, x])

    def _run_vroom(
        self,
        i_residual: np.ndarray,
        q_residual: np.ndarray,
        u_residual: np.ndarray,
        y: int,
        x: int,
    ):
        """
        VROOM-SBI inference at pixel (y, x).

        Returns (result, q_norm, u_norm) where q_norm/u_norm are the
        noiseless VROOM-predicted Q/I and U/I model spectra in cube channel order.

        What gets subtracted for Q at channel c:
            ΔQ[c] = q_norm[c] × I_amplitude × PSF[c]
        where q_norm[c] encodes the full Faraday rotation pattern:
            q_norm[c] = amp × cos(2(χ₀ + RM × λ²[c]))
        """
        from src.simulator.base_simulator import RMSimulator

        i_spec = i_residual[:, y, x].copy()
        q_spec = q_residual[:, y, x].copy()
        u_spec = u_residual[:, y, x].copy()

        if self.freq_sort_idx is not None:
            i_spec = i_spec[self.freq_sort_idx]
            q_spec = q_spec[self.freq_sort_idx]
            u_spec = u_spec[self.freq_sort_idx]

        valid  = (i_spec > 0) & np.isfinite(i_spec)
        q_norm = np.nan_to_num(np.where(valid, q_spec / i_spec, 0.0))
        u_norm = np.nan_to_num(np.where(valid, u_spec / i_spec, 0.0))

        # VROOM inference: posterior p(θ | Q/I, U/I)
        best_result, _ = self.engine.infer(
            np.concatenate([q_norm, u_norm]), n_samples=self.n_samples
        )

        # Posterior mean → noiseless model spectra on cube frequencies
        theta_mean = np.mean(best_result.all_samples, axis=0)

        sim = RMSimulator.__new__(RMSimulator)
        sim.freq             = self.freq_hz.copy()
        sim.lambda_sq        = self.cube_lambda_sq.copy()
        sim._n_freq          = self.n_freq
        sim._weights         = np.ones(self.n_freq)
        sim.n_components     = best_result.n_components
        sim.model_type       = best_result.model_type
        sim._params_per_comp = 3 if sim.model_type == "faraday_thin" else 4
        sim._n_params        = sim._params_per_comp * sim.n_components

        qu_model = sim.simulate_noiseless(theta_mean)
        q_mod    = qu_model[: self.n_freq]
        u_mod    = qu_model[self.n_freq :]

        # Undo frequency reordering
        if self.freq_sort_idx is not None:
            inv   = np.argsort(self.freq_sort_idx)
            q_mod = q_mod[inv]
            u_mod = u_mod[inv]

        return best_result, q_mod, u_mod
