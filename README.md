# APCLEAN — Amortized Polarization CLEAN

A wideband radio interferometric deconvolver that combines:

- **Major cycle** — `casatasks.tclean(niter=0)` for visibility prediction, PSF computation, and residual imaging
- **Minor cycle** — [VROOM-SBI](https://github.com/arpan-das-astrophysics/vroom-sbi) for polarized pixels (Faraday rotation modelling via simulation-based inference) + standard Hogbom for Stokes I-only pixels

The key idea mirrors MTMFS (which fits Taylor polynomials for the I spectrum), but for Q/U: instead of a simple RM-synthesis step, APCLEAN fits a full Faraday rotation model per pixel using VROOM-SBI posterior inference, then subtracts the frequency-dependent Q/U PSF footprint.

---

## Installation

VROOM-SBI must be installed first (in the same environment):

```bash
pip install -e /path/to/vroom-sbi
```

Then install APCLEAN:

```bash
pip install -e /path/to/APCLEAN
```

Requires Python 3.10+, CASA 6.7+ (`casatools`, `casatasks`), and a VROOM-SBI model directory with trained posteriors.

---

## Quick start

**Python API:**

```python
from apclean import APClean

cleaner = APClean(
    ms              = "obs.ms",
    imagename       = "work/apclean",
    imsize          = 256,
    cell            = "15arcsec",
    vroom_model_dir = "models/",
)
cleaner.run(output_prefix="apclean_out", debug=True)
```

**CLI:**

```bash
apclean \
  --ms obs.ms \
  --imagename work/apclean \
  --imsize 256 \
  --cell 15arcsec \
  --vroom-model-dir models/ \
  --output-prefix apclean_out \
  --debug
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ms` | — | Measurement set path |
| `imagename` | — | tclean working image prefix |
| `imsize` | — | Square image size (pixels) |
| `cell` | — | Cell size string, e.g. `15arcsec` |
| `vroom_model_dir` | `models` | Directory with trained VROOM-SBI posteriors (`posterior_*.pt`, `classifier.pt`) |
| `reffreq` | `1.5GHz` | Reference frequency for tclean |
| `robust` | `0.5` | Briggs robust weighting |
| `loop_gain` | `0.1` | Minor-cycle loop gain |
| `threshold_sigma` | `3.0` | Global stopping threshold (× σ_I) |
| `p_snr_threshold` | `5.0` | Polarization detection threshold (× σ_P) |
| `cyclefactor` | `1.5` | Major-cycle trigger: stop minor cycle when peak < cyclefactor × sidelobe_level × P0 |
| `sidelobe_radius` | `15` | Pixels around PSF peak excluded from sidelobe estimation |
| `niter_minor_max` | `5000` | Hard cap on minor-cycle CLEAN iterations per major cycle |
| `max_major` | `10` | Maximum number of major cycles |
| `n_samples` | `1000` | VROOM-SBI posterior samples per polarized pixel |
| `device` | `cuda` | PyTorch device (`cuda` or `cpu`) |

---

## Outputs

All outputs are written to `{output_prefix}_*.fits`:

| File | Contents |
|------|----------|
| `restored_I/Q/U.fits` | Restored channel cubes (model ⊛ clean beam + residual) |
| `model_I/Q/U.fits` | CLEAN model channel cubes |
| `residual_I/Q/U.fits` | Final residual channel cubes |
| `mfs_I/Q/U/P.fits` | σ_I-weighted MFS collapsed 2D images |
| `mfs_frac_pol.fits` | Fractional polarization P/I (2D) |
| `rm.fits` | VROOM-SBI RM map (rad/m²) |
| `p.fits` | VROOM-SBI polarized amplitude map |
| `chi0.fits` | VROOM-SBI intrinsic polarization angle map |
| `ncomp.fits` | Number of Faraday components per pixel |
| `sigma_phi.fits` | Faraday depth width (Burn slab, if fitted) |
| `delta_phi.fits` | Faraday depth separation (two-component, if fitted) |
| `{prefix}.log` | Iteration-by-iteration CLEAN log |
| `{prefix}_diagnostic.pdf` | 6-page diagnostic PDF (`--debug` only) |

---

## Diagnostic PDF (`--debug`)

| Page | Contents |
|------|----------|
| 1 | Convergence — log peak vs cumulative iterations, major-cycle boundaries, CLEAN iterations bar chart |
| 2 | I MFS residual evolution — dirty image through each major cycle to final |
| 3 | Q / U MFS residuals — dirty vs final |
| 4 | Final 2D products — restored I, Q, U, P, fractional polarization + run parameters |
| 5 | VROOM spectra at source pixel — dirty Q/I, U/I vs VROOM model, PA vs λ² |
| 6 | Minor-cycle peak traces — per major cycle |

---

## How it works

```
for each major cycle:
    tclean(niter=0, calcres=True)          ← predict model visibilities, grid residuals
    read residual + PSF from CASA images
    estimate σ_I, σ_P, P map
    compute cycle_thresh = cyclefactor × sidelobe_level × peak

    for each minor-cycle iteration:
        find peak pixel (y, x) in I residual
        if P(y,x) > p_snr_threshold × σ_P:
            run VROOM-SBI → posterior p(RM, χ₀, amp | Q/I, U/I)
            θ_mean → RMSimulator → q_norm[ν], u_norm[ν]
            subtract amp × PSF[ν] from I residual at (y,x)  ← flat spectrum
            subtract q_norm[ν] × amp × PSF[ν] from Q residual
            subtract u_norm[ν] × amp × PSF[ν] from U residual
        else:
            subtract amp × PSF[ν] from I residual only   (Hogbom)
        if peak < cycle_thresh  → trigger next major cycle
        if peak < global_thresh → converged

    write accumulated I/Q/U model back to CASA .model image
```

The subtracted Q/U model encodes the full Faraday rotation:
```
q_norm[ν] = amp × cos(2(χ₀ + RM × λ²[ν]))
u_norm[ν] = amp × sin(2(χ₀ + RM × λ²[ν]))
```

---

## Dependencies

- `numpy`, `scipy`, `astropy`, `matplotlib`
- `casatools >= 6.7`, `casatasks >= 6.7`
- `vroom-sbi` (must be installed as editable package)

---

## License

MIT
