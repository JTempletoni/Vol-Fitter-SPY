# vol_surface_fitter

SVI/SSVI/xSSVI volatility surface fitting and historical event-study analysis for SPY options 2010-2023.

## Status

- **Phase 1** - SVI per-slice fitting via Zeliade quasi-explicit: complete
- **Phase 2** - SSVI/xSSVI surface, BS2002 de-Americanisation, historical batch across 3,500 dates, Bergomi-style event-time decomposition, regime-stratified event study: complete
- **Phase 3** - C++ port of the BS2002 inversion hot path via pybind11: planned
- **Phase 4** - SSR-calibrated generative surface model for synthetic stress scenarios: planned

## Overview

The pipeline reads SPY option chains, de-Americanises quotes via Bjerksund-Stensland 2002 with Newton warm starts and put-call symmetry for puts, fits SVI per slice with an equity prior, escalates to SSVI/xSSVI for the full surface, and runs an event-time decomposition of the observed ATM total variance term structure into a diffusive baseline plus scheduled-event premia.

The historical batch covers 3,500 trading days (2010-2023). Per-snapshot xSSVI parameters and event decompositions are persisted as year-partitioned parquet, with per-date atomic checkpointing for disconnect resilience.

## Repository layout

```
src/volsurf/            Python package
  paths.py              canonical directory paths
  data.py               OptionsDX loader
  rates.py              FRED treasury curve interpolation
  pricing.py            Black-Scholes, BS2002, Leisen-Reimer, de-Am pipeline
  parametric.py         SVI slice model, Zeliade calibration
  parametric_ssvi.py    SSVI, xSSVI, calibration, arbitrage checks
notebooks/              Development notebooks 01-05
  Vol_fitter_01a.ipynb  Data ingest and treasury curve
  Vol_fitter_02.ipynb   BS2002 de-Americanisation, Leisen-Reimer validation
  Vol_fitter_03.ipynb   SVI per-slice fitting, calibration pathologies
  Vol_fitter_04.ipynb   SSVI and xSSVI surface fitting
  Vol_fitter_05.ipynb   Historical refit, event-time decomposition, event study
```

## Reproducing

The notebooks were developed in Google Colab against Google Drive-mounted data. To reproduce locally:

1. Clone this repository.
2. Install the package in editable mode:
```
   pip install -e .
```
3. Download the OptionsDX SPY 2010-2023 corpus from Kaggle (linked below) and stage it under `data_SPY/`.
4. Adjust `src/volsurf/paths.py` if your data lives elsewhere - the module defaults to Colab paths.
5. Run notebooks 01a through 05 in order. Each writes intermediate artefacts that subsequent notebooks consume.

## Data

- **Options**: [OptionsDX SPY End-of-Day Options 2010-2023](https://www.kaggle.com/datasets/kylegraupe/options-data-eod-2010-to-2023) on Kaggle, free tier
- **Rates**: FRED Treasury constant-maturity yields (DGS1MO through DGS30), pulled once and cached to parquet

## Headline result

The SPY surface prices FOMC and CPI premia conditionally on monetary policy regime. Median change in front-month ATM implied volatility at t=+2 trading days after event announcements, computed from the full 2010-2023 event study:

| Regime | FOMC (vol-pts) | CPI (vol-pts) |
|---|---|---|
| 2010-2015 ZIRP | ~0 | ~0 |
| 2016-2019 normalisation | small negative | small negative |
| 2020-2023 COVID/hiking | -2.9 | -1.4 |

Full charts and quantification in notebook 05.

## Methods

- **De-Americanisation**: Bjerksund-Stensland 2002 with put-call symmetry, Newton with OptionsDX warm starts, Brent fallback. Locally-vectorised inversion runs 6-7x faster than the scalar reference.
- **SVI**: Zeliade quasi-explicit calibration with equity prior (rho <= 0), multi-start to escape local minima.
- **SSVI/xSSVI**: full-surface calibration with butterfly and calendar arbitrage constraints. xSSVI adds per-maturity rho with shared eta and gamma.
- **Event-time decomposition**: Bergomi-style forward variance decomposition, `theta(T) = v_typical^2 * T + n_FOMC(T) * variance_FOMC + n_CPI(T) * variance_CPI`, fit per snapshot via NNLS.
- **Event study**: expiry brackets each event, baseline-normalised ATM IV change relative to t=-10 to -6, regime-stratified across 113 FOMC and 168 CPI dates.

## Dependencies

Python 3.10+. See `pyproject.toml` for the full list. Core: NumPy, pandas, SciPy, PyArrow, joblib.

## License

MIT.
