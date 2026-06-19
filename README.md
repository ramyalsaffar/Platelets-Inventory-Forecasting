# Platelet Demand Forecasting and Inventory Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-time--series-8CAAE6)
![XGBoost](https://img.shields.io/badge/XGBoost-gradient%20boosting-006ACC)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


An end-to-end machine learning pipeline that forecasts daily hospital platelet demand and turns the forecast into a perishable-inventory ordering policy. It is built and validated on realistic synthetic blood-center data, so the whole repository runs with no private data.

The problem in one line: platelets expire fast, so collecting too few causes a shortage and collecting too many causes waste. The goal is to forecast demand accurately and order so that both stay low.

---

## Table of contents

- [Why this project](#why-this-project)
- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Methodology](#methodology)
- [Example output](#example-output)
- [The synthetic data](#the-synthetic-data)
- [Limitations](#limitations)
- [Future work](#future-work)
- [References](#references)
- [License](#license)
- [Suggested GitHub topics](#suggested-github-topics)

---

## Why this project

Platelets are a blood product with a very short shelf life: 5 days in the United States, or up to 7 days when extra bacterial testing is used. A blood center must decide how much to collect each day without knowing exactly how much hospitals will transfuse. Two costs pull in opposite directions:

- Collect too little, and a patient does not get a unit. That is a **shortage**.
- Collect too much, and units expire on the shelf. That is **waste**.

This pipeline forecasts demand one day ahead, then runs an ordering policy that keeps both shortage and waste low while respecting the short shelf life. It mirrors the platelet inventory problem studied at Stanford Blood Center and calibrates its synthetic data to a published benchmark from that setting (Guan et al., 2017).

## What it does

The pipeline runs in seven stages, plus a driver that executes them:

1. **Generate** realistic synthetic daily platelet data (collections, transfusions, expiries, stock by age, hospital census, lab signals).
2. **Clean** the data: repair clear problems, flag ambiguous ones, recover censored demand on shortage days, and report data-quality diagnostics.
3. **Build features** that are safe from data leakage (only information known before the day being predicted).
4. **Forecast** daily demand with several model families and pick a winner by accuracy on unseen data.
5. **Tune** the models with a clean train, validation, and test split and documented leakage fixes.
6. **Optimize inventory** with an order-up-to policy and simulate it day by day, including a 5-day and 7-day shelf-life mix.
7. **Plot** the drivers of demand, the forecast against actuals, and the inventory behavior.

Three parts of the pipeline run automatically, with no manual steps: data cleaning (a single function call), model selection (the back-test scores every model and picks the winner), and hyperparameter tuning (a search that tunes each model and keeps a tuned setting only when it beats the default).

## Tech stack

- Language: Python 3.9+
- Data handling: numpy, pandas
- Classical time series (statsmodels): Holt-Winters exponential smoothing, SARIMAX, Theta
- Machine learning (scikit-learn): ElasticNet, Random Forest, cross-validation, and feature scaling
- Gradient boosting: XGBoost and LightGBM
- Inventory math: scipy (the service-level multiplier)
- Charts: matplotlib

## Repository structure

The files are numbered in run order. They are written as sequential steps that share one namespace (like notebook cells), not as importable modules. See [How to run](#how-to-run).

| Order | File | What it contains |
|------:|------|------------------|
| 1 | `1- synthetic_sbc.py` | Synthetic data generator, calibrated to Guan et al. 2017. Also keeps the original generator at the bottom, clearly marked as legacy. |
| 2 | `2- cleaning.py` | `clean_platelet_data` plus the data dictionary. De-censors shortage-day demand, repairs and flags issues, and runs stationarity and seasonality diagnostics. |
| 3 | `3- features.py` | `build_features`. Turns the cleaned table into the leakage-safe feature matrix the models use. |
| 4 | `4- forecasters.py` | The forecasting models and the baseline model-selection back-test. |
| 5 | `5- tuning.py` | Leak-fixed, tuned model variants and the final tuned selection. |
| 6 | `6- inventory_policy.py` | The order-up-to policy and the day-by-day inventory simulation (5-day and 7-day pools). |
| 7 | `7- plotting.py` | Charts: feature drivers, waste and shortage sensitivity, and daily stock by pool. |
| 8 | `8- run_pipeline.py` | Driver. Calls the functions above and prints and plots every stage. |

## Requirements

- Python 3.9 or newer
- `numpy`, `pandas`, `scipy`
- `scikit-learn`
- `statsmodels`
- `xgboost`
- `matplotlib`
- `lightgbm`
- `shap`

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy pandas scipy scikit-learn statsmodels xgboost lightgbm matplotlib shap
```

## How to run

These files are designed to run **in order in a single shared namespace**, the same way notebook cells share state. They are not meant to be imported, so file 4 expects `build_features` from file 3 to already be defined.

**Option A: Jupyter (simplest).** Paste the contents of files 1 through 7 into cells in order and run them, then paste and run the body of `8- run_pipeline.py`. All output appears inline.

**Option B: Spyder.** Turn on one setting first, or you will get a `NameError`:

1. Tools, Preferences, Run, check **"Run in console's namespace instead of an empty one"**, click OK, restart the console.
2. Run files 1 through 7 in order in that console.
3. Run `8- run_pipeline.py`.

`8- run_pipeline.py` prints the cleaning report, the model tables, the inventory comparison and service-level table, and draws four charts. The tuning stage fits many models on five years of data and can take a few minutes; the file shows how to pass smaller settings for a quick run.

## Methodology

### Data cleaning

`clean_platelet_data` repairs only what is unambiguous, flags what needs human judgment, and never lets future information leak into the past. The steps:

- **Dates:** parse to a calendar, drop unparseable rows, remove duplicate dates (counting any that conflict), and insert missing calendar days so every day has exactly one row.
- **Numbers:** coerce count columns to numbers and treat impossible negatives as missing.
- **Short gaps:** fill only gaps of up to two missing days. Target columns are carried forward (past-only, no leakage); driver columns are interpolated. Longer gaps are left for human review.
- **High-side outliers:** flag impossibly large values (many times a column's 99th percentile), such as a mistyped 999999, without touching genuine spikes.
- **Calendar:** recompute day of week, weekend, and US federal holiday from the date.
- **Inventory check:** confirm received minus used minus expired equals the change in total stock, and flag mismatches.
- **De-censor demand:** on shortage days, units used understates need, so the forecast target becomes the larger of used and requested. This is a lower bound on true demand, and the day is flagged.
- **Demand outliers:** flag rare surge days with a robust per-weekday score (median and median absolute deviation), never removing them.
- **Diagnostics (report only):** the ADF stationarity test and STL seasonal and trend strength. ADF = Augmented Dickey-Fuller; STL = Seasonal-Trend decomposition using Loess.

### Forecasting models and why this mix

The stack spans complementary model families, so the best approach is decided by accuracy on unseen data rather than assumed up front:

- **Seasonal-naive:** predicts the same weekday last week. It is the baseline every other model must beat, and it sets the MASE scale.
- **Holt-Winters and Theta:** classical smoothing and decomposition methods, strong on trend plus a weekly season.
- **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous inputs):** a seasonal ARIMA that also uses known drivers (holidays, scheduled surgeries, and lagged hospital census and lab counts).
- **ElasticNet:** a regularized linear model that is interpretable (signed coefficients) and stable when features are correlated.
- **XGBoost, Random Forest, and LightGBM:** tree ensembles that capture nonlinear effects and interactions among the lag, calendar, and clinical features.
- **Trimmed-average ensemble:** averages the models that beat the baseline, which is usually steadier than any single model.

Evaluation is a rolling-origin, one-step-ahead back-test: the model is retrained as time moves forward and only ever predicts the next single day, which mirrors daily operation and avoids look-ahead. Accuracy is scored with MASE (Mean Absolute Scaled Error; below 1 beats the seasonal-naive baseline), with MAE (Mean Absolute Error, in units) alongside. Every feature uses only past information; same-day signals such as census and lab counts are lagged by one day, while calendar effects and scheduled surgeries are known ahead and used directly.

### Stateful model classes

Each model is a class with a streaming interface (fit once, then `predict_step` and `observe` day by day), not a stateless function. This is deliberate, because the models carry state across days. SARIMAX fits once and rolls a fitted state forward with a cheap daily update, re-estimating its coefficients only every 28 days; Holt-Winters caches its smoothing weights and re-optimizes them only every 14 days; the tree models refit on a fixed cadence. A class holds that state cleanly. Each class inherits scikit-learn's BaseEstimator so the tuner can read and set its parameters, with no extra dependency.

### Hyperparameter tuning

Tuning is automated and scored on a held-out validation block; the test block is scored once at the end, and a tuned setting is adopted only when it beats the model's default on validation. Two search methods are used, matched to each model's search space:

- **Random search** for the models with large, continuous spaces: XGBoost, Random Forest, and LightGBM. Random search explores such spaces more efficiently than a full grid (Bergstra and Bengio, 2012), and the learning rate and regularization terms are sampled on a log scale because they act multiplicatively.
- **Grid search** for the models with a small, discrete set of sensible options: SARIMAX (a fixed list of ARIMA orders), Holt-Winters (trend, seasonal, and damping combinations), and Theta (four options). Enumerating these is cheap and exhaustive.
- **ElasticNet** tunes itself through built-in cross-validation over its alpha and l1-ratio grid, using TimeSeriesSplit so folds never train on the future.

Note: tuning here means hyperparameter optimization (choosing model settings), not fine-tuning a pretrained model. ElasticNet uses TimeSeriesSplit (past-only folds) in both the baseline and tuned runs, since shuffling is invalid for time-ordered data. The tuned step adds two SARIMAX leakage fixes: training-only scaling of the drivers (recomputed past-only at each re-estimation) and filling SARIMAX lag gaps with the training mean instead of a future value. It also reports, for comparison, how much a shuffled-fold ElasticNet would have leaked.

### Inventory policy

- **Order-up-to (base-stock) policy:** each day, order enough to bring the usable stock plus the units already in testing up to a target level. The target is expected demand over the protection interval plus a safety buffer.
- **Safety buffer from measured error:** the buffer is sized from the model's actual forecast error over the lead-time-plus-review interval, not a guess.
- **Perishability cap:** the target is capped so the policy does not hold more than can realistically be used before it expires. The cap scales with the shelf-life mix.
- **Dynamic lead time:** routine collection takes a lead time to become usable. If a sudden surge would break a hard safety floor, the policy calls a same-day emergency courier instead of letting a shortage happen.
- **Two-pool perishable model:** the simulation tracks a 5-day pool and a 7-day pool separately, ages each to its own expiry, and issues closest-to-expiry first to minimize waste. A `frac_7day` setting controls the share of arriving units that are 7-day, so you can stress-test the mix.

## Example output

Illustrative figures from the synthetic test window with default settings and a fixed random seed. Exact values depend on settings and the seed.

- Every model beats the seasonal-naive baseline (MASE around 0.96). The best models land near MASE 0.6, with MAE close to 4 to 5 platelet units. The ensemble is competitive with the best single model.
- The cleaning step recovers demand on censored shortage days, flags demand spikes, and confirms the series is stationary with a clear weekly pattern.
- The order-up-to policy meets demand on every simulated day through the safety floor plus the emergency courier, while keeping waste and on-shelf stock low. A service-level table shows the trade-off between waste and how often a courier is needed.

The pipeline also produces four charts: feature drivers of demand, forecast against actuals over the back-test window, waste and shortage sensitivity to the 7-day share, and daily stock split into the 5-day and 7-day pools.

The drivers chart adapts to the winning model. An ElasticNet winner shows signed coefficients (which features raise or lower demand). A tree winner (XGBoost, Random Forest, or LightGBM) shows SHAP average impact (SHAP, SHapley Additive exPlanations, splits each prediction into per-feature contributions), with a fallback to the model's built-in feature importances if the shap package is not installed. A time-series winner shows a companion ElasticNet, fitted only to make the drivers explicit.

## The synthetic data

- The data is fully synthetic and reproducible with a fixed seed. No real or private data is used.
- It is calibrated to Guan et al. (2017), a Stanford Blood Center platelet study, so weekday and weekend demand levels, the seasonal pattern, and the baseline waste rate are realistic.
- The daily table includes platelet flow (used, requested, received, expired), stock by remaining shelf life, hospital census (including the cancer ward, the main platelet driver), scheduled surgeries, and abnormal CBC (Complete Blood Count) lab counts.
- Acronyms used in the data: CBC = Complete Blood Count; ICU = Intensive Care Unit; MCV = Mean Corpuscular Volume; RDW = Red cell Distribution Width.

## Limitations

- Results come from synthetic data. Real blood-center data should drive any final numbers.
- Near-zero waste is idealized; a real center sees a few percent from collection logistics.
- Very large surges still rely on the emergency courier being available the same day.
- The model forecasts one day ahead and covers one product at one site. Multi-day horizons and sharing across hospitals are noted as extensions, not implemented.

## Future work

The largest planned extension is multi-day forecasting. The pipeline currently predicts demand one day ahead, and the ordering policy reuses that single number across the protection interval. A short multi-day forecast would match how the perishable-inventory literature operates: the benchmark this project calibrates to forecasts several days ahead (Guan et al., 2017), and related modeling studies forecast two to four days ahead. The useful horizon is short, roughly the order lead time plus a few days of coverage, and it must stay inside the five-to-seven-day shelf life. About three to five days is a sensible target: long enough to cover a one-to-two-day lead time plus a small buffer, but short enough that the units can still be used before they expire. Predicting much further ahead adds little value, because platelets cannot be stored long.

Changes this would require:

- **Statistical models** (seasonal-naive, Holt-Winters, SARIMAX, Theta) can forecast several days in one call, so these are quick. SARIMAX additionally needs the driver values for the forecast days; calendar effects and scheduled surgeries are known ahead, but the lagged census and lab features are not, so they would be dropped over the horizon or forecast separately.
- **Machine-learning models** (XGBoost, Random Forest, LightGBM, ElasticNet) predict a single step from lag features, so they need a multi-step strategy: either recursive (feed each day's prediction back in as the next day's lag) or direct (train one model per day-ahead, for example one model each for days one through five). Both also need the future feature values handled as above.
- **Evaluation** would expand: the rolling-origin back-test and the MASE score would report accuracy for each horizon (day one through day H) instead of a single day.
- **Inventory side** needs little change. The order-up-to policy already accepts a per-day forecast list, so it would receive real per-day forecasts instead of one number repeated, which should make the ordering decision more accurate.

## References

- Guan, L., et al. (2017). Big data modeling to predict platelet usage and minimize wastage in a tertiary care system. *Proceedings of the National Academy of Sciences (PNAS)*, 114(43), 11368-11373. The calibration target and the closest published benchmark.
- Hyndman, R. J., and Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. Source of MASE.
- Bergmeir, C., and Benitez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213. Basis for rolling-origin and TimeSeriesSplit validation.
- Bergstra, J., and Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research (JMLR)*, 13, 281-305. Basis for the random hyperparameter search.
- Fontaine, M. J., et al. (2009). Improving platelet supply chains through collaborations between blood centers and transfusion services. *Transfusion*, 49(10), 2040-2047. A Stanford Blood Center precedent for cutting platelet outdate.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Suggested GitHub topics

Add these as repository topics (Settings, then the gear next to About) so the project is easy to find:

`machine-learning` `time-series-forecasting` `demand-forecasting` `inventory-optimization` `operations-research` `perishable-inventory` `healthcare` `blood-bank` `python` `scikit-learn` `xgboost` `statsmodels` `forecasting` `supply-chain`
