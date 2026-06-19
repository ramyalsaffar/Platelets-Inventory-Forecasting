"""
Leak fixes + automated hyperparameter tuning, built on top of forecasters.py.

Run this AFTER forecasters.py and features.py. It uses these names defined
earlier: BaseForecaster, backtest_model, SeasonalNaiveForecaster, ElasticNetForecaster,
SARIMAX_EXOG_COLS, build_features.

WHAT THIS DOES
  1) Fixes three look-ahead leaks (numbers change on purpose):
       a. SARIMAX exogenous scaling: standardize with TRAIN-ONLY statistics, recomputed
          past-only at each re-estimation, instead of full-series statistics.
       b. SARIMAX lagged columns: fill the leading gap with the training-period mean, not a
          backward fill that copies a future value.
       c. ElasticNet cross-validation: use TimeSeriesSplit (past-only folds) instead of the
          default KFold, whose first fold trains on later days to score earlier days.
  2) Tunes hyperparameters with a dependency-free random/grid search scored on a held-out
     VALIDATION block, adopting a tuned configuration only if it beats the model's default on
     validation. Optuna is NOT required.
  3) Splits the series by PERCENTAGE (test_frac, val_frac; train is the remainder), so it
     rescales to any dataset length.

The final evaluation runs once on the TEST block. The validation block is used only to choose
hyperparameters and is never used to report the final numbers.
"""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

# LightGBM is a standard model in the stack; it is imported in forecasters.py (run first).

SEED = 42


@dataclass
class TuningContext:
    """Like the back-test context, but carries the RAW (unstandardized) exogenous matrix so the
    SARIMAX model can standardize it with training-only statistics."""
    y: np.ndarray
    Xv: np.ndarray
    exog_raw: np.ndarray
    warmup: int
    m: int
    t0: int
    n: int


# --------------------------------------------------------------------------------------------
# Leak-fixed and tunable model variants
# --------------------------------------------------------------------------------------------
class SARIMAXTrainScaled(BaseForecaster):
    """SARIMAX whose exogenous regressors are standardized using TRAINING-ONLY statistics,
    recomputed past-only at each full re-estimation. This removes the full-series scaling leak.
    The order and seasonal_order are tunable."""
    name = "sarimax"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), reestimate_every: int = 28):
        """Store the ARIMA order, seasonal order, and re-estimation cadence."""
        self.order = order
        self.seasonal_order = seasonal_order
        self.reestimate_every = reestimate_every

    def _standardize(self, raw_rows):
        """Standardize raw exogenous rows using the training-only mean and standard deviation."""
        return (raw_rows - self.mu_) / self.sd_

    def _fit_full(self, upto: int):
        """Recompute the training-only standardization on data up to `upto`, then fit the full SARIMAX model on the standardized exogenous matrix. This removes the full-series scaling leak."""
        train_raw = self.ctx_.exog_raw[:upto]
        self.mu_ = train_raw.mean(axis=0)
        sd = train_raw.std(axis=0)
        sd[sd == 0] = 1.0                      # guard against a constant column
        self.sd_ = sd
        exs_train = (train_raw - self.mu_) / self.sd_
        return SARIMAX(self.ctx_.y[:upto], exog=exs_train, order=self.order,
                       seasonal_order=self.seasonal_order, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=0)

    def fit(self, ctx):
        """Fit the model once on the training history (up to the first test day)."""
        self.ctx_ = ctx
        self.res_ = self._fit_full(ctx.t0)
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t from the current fitted state, standardizing that day's exogenous row with the training-only statistics."""
        check_is_fitted(self, "res_")
        row = self._standardize(self.ctx_.exog_raw[t:t + 1])
        return float(self.res_.forecast(1, exog=row)[0])

    def observe(self, t: int) -> None:
        """Append day t's actual and standardized exogenous row to the state (no refit), and re-estimate the coefficients every reestimate_every days."""
        check_is_fitted(self, "res_")
        ctx = self.ctx_
        i = t - ctx.t0
        row = self._standardize(ctx.exog_raw[t:t + 1])
        self.res_ = self.res_.append(ctx.y[t:t + 1], exog=row, refit=False)
        if (i + 1) % self.reestimate_every == 0 and t + 1 < ctx.n:
            self.res_ = self._fit_full(t + 1)


class HoltWintersTunable(BaseForecaster):
    """Holt-Winters with tunable trend (additive or none), seasonal (additive or multiplicative),
    and damping. With trend='add', seasonal='add', damped=False it matches the base Holt-Winters model."""
    name = "holt_winters"

    def __init__(self, season_length: int = 7, refit_every: int = 14,
                 trend="add", seasonal="add", damped: bool = False, init_method: str = "heuristic"):
        """Store the smoothing settings (trend, seasonal, damping) and the refit cadence."""
        self.season_length = season_length
        self.refit_every = refit_every
        self.trend = trend
        self.seasonal = seasonal
        self.damped = damped
        self.init_method = init_method

    def _make(self, series):
        """Build an unfitted ExponentialSmoothing model on the series with the configured trend, season, and optional damping."""
        return ExponentialSmoothing(
            series, trend=self.trend, seasonal=self.seasonal,
            damped_trend=(self.damped and self.trend is not None),
            seasonal_periods=self.season_length, initialization_method=self.init_method)

    def fit(self, ctx):
        """Keep the context and clear any cached smoothing weights."""
        self.ctx_ = ctx
        self.weights_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days re-optimize the smoothing weights on the history; in between reuse them and refit only the levels."""
        check_is_fitted(self, "ctx_")
        y = self.ctx_.y
        i = t - self.ctx_.t0
        if i % self.refit_every == 0:
            self.weights_ = dict(self._make(y[:t]).fit().params)
        fixed = {k: self.weights_[k] for k in
                 ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend")
                 if k in self.weights_ and self.weights_[k] is not None and not np.isnan(self.weights_[k])}
        f = self._make(y[:t]).fit(optimized=False, **fixed)
        return float(f.forecast(1)[0])


class XGBoostTunable(BaseForecaster):
    """XGBoost with the full tunable set, including min_child_weight and reg_alpha. With the
    default arguments it matches the base XGBoost model."""
    name = "xgboost"

    def __init__(self, refit_every: int = 14, n_estimators: int = 300, max_depth: int = 3,
                 learning_rate: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, reg_alpha: float = 0.0, min_child_weight: float = 1.0,
                 random_state: int = SEED):
        """Store the full XGBoost hyperparameter set (including reg_alpha and min_child_weight) and the refit cadence."""
        self.refit_every = refit_every
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.min_child_weight = min_child_weight
        self.random_state = random_state

    def fit(self, ctx):
        """Keep the context and clear the cached model so it refits on the first prediction."""
        self.ctx_ = ctx
        self.model_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days refit the booster on all data from warmup up to day t, then predict from that day's features."""
        check_is_fitted(self, "ctx_")
        ctx = self.ctx_
        i = t - ctx.t0
        if i % self.refit_every == 0:
            tr = slice(ctx.warmup, t)
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha, min_child_weight=self.min_child_weight,
                random_state=self.random_state, objective="reg:squarederror")
            self.model_.fit(ctx.Xv[tr], ctx.y[tr])
        return float(self.model_.predict(ctx.Xv[t:t + 1])[0])


# --------------------------------------------------------------------------------------------
# Search spaces and scoring
# --------------------------------------------------------------------------------------------
SARIMAX_DEFAULT = ((1, 1, 1), (1, 0, 1, 7))
SARIMAX_CANDIDATES = [
    ((1, 1, 1), (1, 0, 1, 7)),   # current default (always included)
    ((0, 1, 1), (0, 0, 1, 7)),
    ((1, 1, 0), (1, 0, 0, 7)),
    ((2, 1, 1), (1, 0, 1, 7)),
    ((1, 1, 2), (1, 0, 1, 7)),
    ((1, 1, 1), (0, 0, 1, 7)),
    ((1, 1, 1), (1, 0, 0, 7)),
    ((2, 1, 2), (1, 0, 1, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
    ((0, 1, 1), (0, 1, 1, 7)),
]
HW_DEFAULT = dict(trend="add", seasonal="add", damped=False)


def _hw_grid(allow_mul: bool):
    """Build the Holt-Winters search grid over trend, seasonal, and damping. Multiplicative seasonality is included only when allow_mul is True, since it needs strictly positive data."""
    grid = []
    for trend in ("add", None):
        for seasonal in ("add", "mul"):
            for damped in (False, True):
                if trend is None and damped:
                    continue
                if seasonal == "mul" and not allow_mul:
                    continue
                grid.append(dict(trend=trend, seasonal=seasonal, damped=damped))
    return grid


def _sample_xgb(rng):
    """Draw one random XGBoost hyperparameter set (log-uniform for the learning rate and the two regularization terms)."""
    lr  = 10 ** rng.uniform(np.log10(0.01), np.log10(0.3))
    lam = 10 ** rng.uniform(np.log10(1e-3), np.log10(10.0))
    alp = 10 ** rng.uniform(np.log10(1e-3), np.log10(10.0))
    return dict(
        learning_rate=float(lr),
        max_depth=int(rng.integers(3, 11)),
        min_child_weight=int(rng.integers(1, 11)),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.6, 1.0)),
        reg_lambda=float(lam),
        reg_alpha=float(alp),
        n_estimators=int(rng.integers(200, 801)),
    )


def _sample_rf(rng):
    """Draw one random random-forest hyperparameter set."""
    return dict(
        n_estimators=int(rng.integers(200, 801)),
        max_depth=int(rng.integers(3, 21)),
        max_features=float(rng.uniform(0.3, 1.0)),
        min_samples_leaf=int(rng.integers(1, 11)),
    )


def _sample_lgb(rng):
    """Draw one random LightGBM hyperparameter set (log-uniform for the learning rate and L2 term)."""
    lr  = 10 ** rng.uniform(np.log10(0.01), np.log10(0.3))
    lam = 10 ** rng.uniform(np.log10(1e-3), np.log10(10.0))
    return dict(
        learning_rate=float(lr),
        num_leaves=int(rng.integers(15, 128)),
        min_child_samples=int(rng.integers(5, 51)),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.6, 1.0)),
        reg_lambda=float(lam),
        n_estimators=int(rng.integers(200, 801)),
    )


THETA_DEFAULT = dict(deseasonalize=True, method="auto")
def _theta_grid():
    """Return the small grid of Theta-method options to try."""
    return [dict(deseasonalize=True, method="auto"),
            dict(deseasonalize=True, method="additive"),
            dict(deseasonalize=True, method="multiplicative"),
            dict(deseasonalize=False, method="auto")]


def _val_mase(model, val_ctx, Q_val, quiet):
    """Back-test the model on the validation block and return its MASE (mean absolute scaled error)."""
    fc = backtest_model(model, val_ctx, suppress_warnings=quiet)
    actual = val_ctx.y[val_ctx.t0:val_ctx.n]
    return float(np.mean(np.abs(actual - fc)) / Q_val)


def tune_xgboost(val_ctx, Q_val, n_trials, seed, refit_every):
    """Random-search XGBoost on the validation block. Return the best config, its validation MASE, the default's MASE, and whether the tuned config beat the default."""
    rng = np.random.default_rng(seed)
    default_mase = _val_mase(XGBoostTunable(refit_every=refit_every, random_state=seed),
                             val_ctx, Q_val, quiet=False)
    best_mase, best_params = default_mase, None
    for _ in range(n_trials):
        p = _sample_xgb(rng)
        mase = _val_mase(XGBoostTunable(refit_every=refit_every, random_state=seed, **p),
                         val_ctx, Q_val, quiet=False)
        if mase < best_mase:
            best_mase, best_params = mase, p
    if best_params is not None:
        return dict(refit_every=refit_every, random_state=seed, **best_params), best_mase, default_mase, True
    return dict(refit_every=refit_every, random_state=seed), default_mase, default_mase, False


def tune_sarimax(val_ctx, Q_val, reestimate_every):
    """Grid-search the SARIMAX order and seasonal order over a fixed candidate list on the validation block. Return the best config, its MASE, the default's MASE, and whether to adopt the tuned one."""
    rows = []
    for order, sorder in SARIMAX_CANDIDATES:
        try:
            mase = _val_mase(SARIMAXTrainScaled(order=order, seasonal_order=sorder,
                                                reestimate_every=reestimate_every),
                             val_ctx, Q_val, quiet=True)
        except Exception:
            mase = float("inf")
        rows.append((mase, order, sorder))
    rows.sort(key=lambda r: r[0])
    best_mase, b_o, b_s = rows[0]
    default_mase = next(mase for mase, o, s in rows if (o, s) == SARIMAX_DEFAULT)
    if (b_o, b_s) != SARIMAX_DEFAULT and best_mase < default_mase:
        return dict(order=b_o, seasonal_order=b_s, reestimate_every=reestimate_every), best_mase, default_mase, True
    return dict(order=SARIMAX_DEFAULT[0], seasonal_order=SARIMAX_DEFAULT[1],
                reestimate_every=reestimate_every), default_mase, default_mase, False


def tune_holtwinters(val_ctx, Q_val, refit_every):
    """Grid-search the Holt-Winters trend, seasonal, and damping options on the validation block. Return the best config, its MASE, the default's MASE, and whether to adopt the tuned one."""
    allow_mul = bool(np.all(val_ctx.y > 0))
    rows = []
    for g in _hw_grid(allow_mul):
        try:
            mase = _val_mase(HoltWintersTunable(refit_every=refit_every, **g), val_ctx, Q_val, quiet=True)
        except Exception:
            mase = float("inf")
        rows.append((mase, g))
    rows.sort(key=lambda r: r[0])
    best_mase, best_g = rows[0]
    default_mase = next(mase for mase, g in rows if g == HW_DEFAULT)
    if best_g != HW_DEFAULT and best_mase < default_mase:
        return dict(refit_every=refit_every, **best_g), best_mase, default_mase, True
    return dict(refit_every=refit_every, **HW_DEFAULT), default_mase, default_mase, False


def tune_random_forest(val_ctx, Q_val, n_trials, seed, refit_every):
    """Random-search the random forest on the validation block (its own random stream). Return the best config, its MASE, the default's MASE, and whether to adopt the tuned one."""
    rng = np.random.default_rng(seed + 1)            # own stream, distinct from XGBoost's
    default_mase = _val_mase(RandomForestForecaster(refit_every=refit_every, random_state=seed),
                             val_ctx, Q_val, quiet=False)
    best_mase, best_params = default_mase, None
    for _ in range(n_trials):
        p = _sample_rf(rng)
        mase = _val_mase(RandomForestForecaster(refit_every=refit_every, random_state=seed, **p),
                         val_ctx, Q_val, quiet=False)
        if mase < best_mase:
            best_mase, best_params = mase, p
    if best_params is not None:
        return dict(refit_every=refit_every, random_state=seed, **best_params), best_mase, default_mase, True
    return dict(refit_every=refit_every, random_state=seed), default_mase, default_mase, False


def tune_lightgbm(val_ctx, Q_val, n_trials, seed, refit_every):
    """Random-search LightGBM on the validation block. Return the best config, its MASE, the default's MASE, and whether to adopt the tuned one."""
    rng = np.random.default_rng(seed + 2)
    default_mase = _val_mase(LightGBMForecaster(refit_every=refit_every, random_state=seed),
                             val_ctx, Q_val, quiet=False)
    best_mase, best_params = default_mase, None
    for _ in range(n_trials):
        p = _sample_lgb(rng)
        mase = _val_mase(LightGBMForecaster(refit_every=refit_every, random_state=seed, **p),
                         val_ctx, Q_val, quiet=False)
        if mase < best_mase:
            best_mase, best_params = mase, p
    if best_params is not None:
        return dict(refit_every=refit_every, random_state=seed, **best_params), best_mase, default_mase, True
    return dict(refit_every=refit_every, random_state=seed), default_mase, default_mase, False


def tune_theta(val_ctx, Q_val, m):
    """Grid-search the Theta-method options on the validation block. Return the best config, its MASE, the default's MASE, and whether to adopt the tuned one."""
    rows = []
    for g in _theta_grid():
        try:
            mase = _val_mase(ThetaForecaster(season_length=m, **g), val_ctx, Q_val, quiet=True)
        except Exception:
            mase = float("inf")
        rows.append((mase, g))
    rows.sort(key=lambda r: r[0])
    best_mase, best_g = rows[0]
    default_mase = next(mase for mase, g in rows if g == THETA_DEFAULT)
    if best_g != THETA_DEFAULT and best_mase < default_mase:
        return dict(season_length=m, **best_g), best_mase, default_mase, True
    return dict(season_length=m, **THETA_DEFAULT), default_mase, default_mase, False


# --------------------------------------------------------------------------------------------
# Main split-aware, leak-fixed, tuned selector
# --------------------------------------------------------------------------------------------
def run_tuned_selection(clean_df, test_frac=0.15, val_frac=0.15, refit_every=14,
                        sarimax_reestimate=28, xgb_trials=40, rf_trials=40, lgb_trials=40,
                        seed=SEED, verbose=True):
    """Percentage-split, leak-fixed, tuned model selection.
    Returns a dict with the test-block scores/forecasts/table, the chosen configurations and
    whether each beat its default on validation, and the old-vs-new ElasticNet test comparison."""
    df = clean_df.reset_index(drop=True)
    y = df["demand_model"].values.astype(float)
    Xdf = build_features(df)
    Xv = Xdf.values
    n = len(y); m = 7; warmup = 14

    test_size = int(round(n * test_frac))
    val_size = int(round(n * val_frac))
    train_size = n - val_size - test_size
    val_start = train_size
    test_start = train_size + val_size
    if train_size < max(35, 2 * m):
        raise ValueError(
            f"Not enough history: train fraction leaves only {train_size} training days, but at "
            f"least {max(35, 2 * m)} are required. Lower test_frac/val_frac or provide more data.")

    # RAW exogenous matrix; lagged columns' leading gap filled with the TRAIN-block mean (no future).
    ex = Xdf[SARIMAX_EXOG_COLS].copy()
    for col in ["census_heme_onc_lag1", "cbc_low_platelet_lag1"]:
        ex[col] = ex[col].fillna(float(ex[col].iloc[:train_size].mean()))
    exog_raw = ex.values

    val_ctx = TuningContext(y=y, Xv=Xv, exog_raw=exog_raw, warmup=warmup, m=m, t0=val_start, n=test_start)
    test_ctx = TuningContext(y=y, Xv=Xv, exog_raw=exog_raw, warmup=warmup, m=m, t0=test_start, n=n)
    Q_val = float(np.mean(np.abs(y[m:val_start] - y[:val_start - m])))
    Q_test = float(np.mean(np.abs(y[m:test_start] - y[:test_start - m])))

    if verbose:
        print(f"Split (percent): train {train_size} ({train_size/n:.0%}), "
              f"validation {val_size} ({val_size/n:.0%}), test {test_size} ({test_size/n:.0%}).")
        print("Tuning on the validation block; the test block is scored once at the end.\n")

    # ---- tune on validation (adopt a tuned config only if it beats the default) ----
    xgb_cfg, xgb_b, xgb_d, xgb_adopt = tune_xgboost(val_ctx, Q_val, xgb_trials, seed, refit_every)
    sx_cfg,  sx_b,  sx_d,  sx_adopt  = tune_sarimax(val_ctx, Q_val, sarimax_reestimate)
    hw_cfg,  hw_b,  hw_d,  hw_adopt  = tune_holtwinters(val_ctx, Q_val, refit_every)
    rf_cfg,  rf_b,  rf_d,  rf_adopt  = tune_random_forest(val_ctx, Q_val, rf_trials, seed, refit_every)
    th_cfg,  th_b,  th_d,  th_adopt  = tune_theta(val_ctx, Q_val, m)
    lgb_cfg, lgb_b, lgb_d, lgb_adopt = tune_lightgbm(val_ctx, Q_val, lgb_trials, seed, refit_every)

    tuning_report = {
        "xgboost":     {"validation_mase_default": round(xgb_d, 4), "validation_mase_best": round(xgb_b, 4),
                        "adopted_tuned": xgb_adopt, "config": xgb_cfg},
        "sarimax":     {"validation_mase_default": round(sx_d, 4),  "validation_mase_best": round(sx_b, 4),
                        "adopted_tuned": sx_adopt,  "config": {k: sx_cfg[k] for k in ("order", "seasonal_order")}},
        "holt_winters":{"validation_mase_default": round(hw_d, 4),  "validation_mase_best": round(hw_b, 4),
                        "adopted_tuned": hw_adopt,  "config": {k: hw_cfg[k] for k in ("trend", "seasonal", "damped")}},
        "random_forest":{"validation_mase_default": round(rf_d, 4), "validation_mase_best": round(rf_b, 4),
                        "adopted_tuned": rf_adopt,  "config": {k: v for k, v in rf_cfg.items() if k not in ("refit_every", "random_state")}},
        "theta":       {"validation_mase_default": round(th_d, 4),  "validation_mase_best": round(th_b, 4),
                        "adopted_tuned": th_adopt,  "config": {k: th_cfg[k] for k in ("deseasonalize", "method")}},
    }
    tuning_report["lightgbm"] = {"validation_mase_default": round(lgb_d, 4), "validation_mase_best": round(lgb_b, 4),
                    "adopted_tuned": lgb_adopt, "config": {k: v for k, v in lgb_cfg.items() if k not in ("refit_every", "random_state")}}

    # ---- build the final models (fresh instances per use, so val and test runs share no state) ----
    def _make_final_models():
        """Build a fresh set of the final models from the chosen configurations, so the validation and test runs never share fitted state."""
        fm = {
            "seasonal_naive": SeasonalNaiveForecaster(season_length=m),
            "holt_winters":   HoltWintersTunable(**hw_cfg),
            "sarimax":        SARIMAXTrainScaled(**sx_cfg),
            "xgboost":        XGBoostTunable(**xgb_cfg),
            "elasticnet":     ElasticNetForecaster(refit_every=refit_every,
                                                   cv=TimeSeriesSplit(n_splits=5), random_state=seed),
            "random_forest":  RandomForestForecaster(**rf_cfg),
            "theta":          ThetaForecaster(**th_cfg),
            "lightgbm":       LightGBMForecaster(**lgb_cfg),
        }
        return fm

    quiet = {"holt_winters", "sarimax", "theta"}
    base = list(_make_final_models().keys())

    # ---- SELECTION on VALIDATION only (the test block is never used to choose anything) ----
    # The ensemble members and the headline winner are picked on validation, so the test numbers
    # below are an honest held-out estimate, not a best-of-N chosen on the data it is reported on.
    fc_val = {name: backtest_model(mdl, val_ctx, suppress_warnings=name in quiet)
              for name, mdl in _make_final_models().items()}
    actual_val = y[val_start:test_start]
    def mase_val(p):
        """MASE of a validation-block forecast against the validation actuals."""
        return float(np.mean(np.abs(actual_val - p)) / Q_val)
    val_scores = {k: mase_val(v) for k, v in fc_val.items()}
    members = [k for k in base if k != "seasonal_naive" and val_scores[k] < val_scores["seasonal_naive"]] or ["seasonal_naive"]
    val_scores["avg_trimmed"] = mase_val(np.mean([fc_val[k] for k in members], axis=0))
    winner = min(val_scores, key=val_scores.get)          # winner decided on validation

    # ---- final evaluation on the TEST block (once), using the validation-chosen members ----
    fc = {name: backtest_model(mdl, test_ctx, suppress_warnings=name in quiet)
          for name, mdl in _make_final_models().items()}
    actual = y[test_start:n]
    def mase(p):
        """MASE of a test-block forecast against the test actuals."""
        return float(np.mean(np.abs(actual - p)) / Q_test)
    scores = {k: mase(v) for k, v in fc.items()}
    fc["avg_trimmed"] = np.mean([fc[k] for k in members], axis=0)   # members fixed from validation
    scores["avg_trimmed"] = mase(fc["avg_trimmed"])


    # ---- one-time leak quantification: old (KFold) vs fixed (TimeSeriesSplit) ElasticNet on TEST ----
    old_en = ElasticNetForecaster(refit_every=refit_every, cv=5, random_state=seed)
    fc_old_en = backtest_model(old_en, test_ctx, suppress_warnings=False)
    en_leak = {"elasticnet_test_mase_old_kfold_leaky": round(mase(fc_old_en), 4),
               "elasticnet_test_mase_fixed_timeseriessplit": round(scores["elasticnet"], 4)}

    table = pd.DataFrame({
        "MASE": pd.Series(scores),
        "MAE":  pd.Series({k: float(np.mean(np.abs(actual - v))) for k, v in fc.items()})
    }).loc[base + ["avg_trimmed"]].round({"MASE": 3, "MAE": 2})

    if verbose:
        for name, info in tuning_report.items():
            tag = "ADOPTED tuned" if info["adopted_tuned"] else "kept default"
            print(f"  {name:13s}: validation MASE default {info['validation_mase_default']} -> "
                  f"best {info['validation_mase_best']}  [{tag}]")
        print("\nTEST-block results (honest held-out; members and winner were chosen on VALIDATION):")
        print(table.to_string())
        print(f"WINNER (chosen on validation): {winner}  ->  test MASE {round(scores[winner], 3)}")
        print("ensemble members (chosen on validation):", members)
        print("ElasticNet leak check on test  -> leaky KFold:", en_leak["elasticnet_test_mase_old_kfold_leaky"],
              "| fixed TimeSeriesSplit:", en_leak["elasticnet_test_mase_fixed_timeseriessplit"])

    return {"scores": scores, "forecasts": fc, "actual": actual,
            "dates": df["date"].values[test_start:n], "winner": winner, "members": members,
            "winner_selected_on": "validation",
            "validation_scores": {k: round(v, 4) for k, v in val_scores.items()},
            "table": table, "tuning_report": tuning_report, "elasticnet_leak_check": en_leak,
            "split": {"train": train_size, "validation": val_size, "test": test_size}}