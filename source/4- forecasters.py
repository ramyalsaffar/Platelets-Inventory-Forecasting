"""
Forecaster classes + rolling-origin back-test engine + model registry + main selector.

Each model is a small stateful class; a rolling-origin back-test scores them and selects the
most accurate model, plus a trimmed-average ensemble, on held-out days.

WHY CLASSES (not bare functions):
  The statistical models carry state from one day to the next. SARIMAX fits once, then rolls a
  fitted state object forward each day with a cheap append, and only re-estimates every 28 days.
  Holt-Winters caches its smoothing weights and re-optimizes them only every 14 days. A class
  holds that state cleanly. Each class inherits scikit-learn's BaseEstimator (only), so
  get_params/set_params work for tuning. No new dependency is added.

INTERFACE every model implements:
  fit(ctx)           initial fit on history up to ctx.t0
  predict_step(t)    one-step-ahead forecast for day t (the model is trained through day t-1)
  observe(t)         ingest the actual for day t (cheap update and the model's own refit cadence)

NOTE on tuning: these models use a streaming fit(ctx)/predict_step(t) interface, NOT the
scikit-learn fit(X, y)/predict(X) contract. They are therefore tuned with a custom loop
(set_params plus the back-test engine), not scikit-learn's GridSearchCV. That is why they
inherit BaseEstimator only and not RegressorMixin.
"""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

SEED = 42  # global random seed for reproducibility


@dataclass
class BacktestContext:
    """Everything the models need during the back-test, computed once by the selector."""
    y: np.ndarray          # de-censored demand (the forecast target)
    Xv: np.ndarray         # feature matrix from build_features
    exs: np.ndarray        # standardized exogenous matrix for SARIMAX
    warmup: int            # first row index the machine-learning models may train from
    m: int                 # seasonal period (7 days)
    t0: int                # first test index
    n: int                 # length of the series


class BaseForecaster(BaseEstimator):
    """Common interface. Subclasses implement predict_step; fit and observe are optional.
    Inherits BaseEstimator only (for get_params and set_params). It deliberately does NOT
    inherit RegressorMixin, because these forecasters use a streaming fit(ctx)/predict_step(t)
    interface rather than the scikit-learn fit(X, y)/predict(X) contract, so RegressorMixin's
    score method would be misleading and non-functional here."""
    name = "base"

    def fit(self, ctx: BacktestContext):
        """Store the back-test context and return self. Models that need an initial estimate (such as SARIMAX) override this."""
        self.ctx_ = ctx
        return self

    def predict_step(self, t: int) -> float:
        """Return the one-step-ahead forecast for day t. Subclasses must implement this; the model is treated as trained through day t-1."""
        raise NotImplementedError

    def observe(self, t: int) -> None:
        """Ingest the actual value for day t. The default does nothing; streaming models override this to update state and honor their refit cadence."""
        pass


class SeasonalNaiveForecaster(BaseForecaster):
    """Predict the same weekday last week. The baseline every other model must beat."""
    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        """Set the seasonal period (7 means the same weekday one week earlier)."""
        self.season_length = season_length

    def predict_step(self, t: int) -> float:
        """Forecast day t as the actual value one season earlier (same weekday last week)."""
        check_is_fitted(self, "ctx_")
        return float(self.ctx_.y[t - self.season_length])


class HoltWintersForecaster(BaseForecaster):
    """Exponential smoothing with a weekly season. Re-optimizes its smoothing weights every
    refit_every days and reuses them in between."""
    name = "holt_winters"

    def __init__(self, season_length: int = 7, refit_every: int = 14,
                 trend: str = "add", seasonal: str = "add", init_method: str = "heuristic"):
        """Store the smoothing settings and how often to re-optimize the weights."""
        self.season_length = season_length
        self.refit_every = refit_every
        self.trend = trend
        self.seasonal = seasonal
        self.init_method = init_method

    def fit(self, ctx):
        """Keep the context and clear any cached smoothing weights."""
        self.ctx_ = ctx
        self.weights_ = None
        return self

    def _make(self, series):
        """Build an unfitted ExponentialSmoothing model on the series with the configured trend and weekly season."""
        return ExponentialSmoothing(series, trend=self.trend, seasonal=self.seasonal,
                                    seasonal_periods=self.season_length,
                                    initialization_method=self.init_method)

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days re-optimize the smoothing weights on the history so far; in between reuse those weights and refit only the levels."""
        check_is_fitted(self, "ctx_")
        y = self.ctx_.y
        i = t - self.ctx_.t0
        if i % self.refit_every == 0:
            f0 = self._make(y[:t]).fit()
            self.weights_ = (f0.params["smoothing_level"], f0.params["smoothing_trend"],
                             f0.params["smoothing_seasonal"])
        a, b, g = self.weights_
        f = self._make(y[:t]).fit(smoothing_level=a, smoothing_trend=b,
                                  smoothing_seasonal=g, optimized=False)
        return float(f.forecast(1)[0])


class SARIMAXForecaster(BaseForecaster):
    """Seasonal ARIMA with exogenous regressors. Fits once, rolls its state forward daily with a
    cheap append, and re-estimates the coefficients every reestimate_every days."""
    name = "sarimax"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), reestimate_every: int = 28):
        """Store the ARIMA order, seasonal order, and how often to re-estimate the coefficients."""
        self.order = order
        self.seasonal_order = seasonal_order
        self.reestimate_every = reestimate_every

    def _fit_full(self, upto: int):
        """Fit the full SARIMAX model on the demand and exogenous data up to index `upto`."""
        return SARIMAX(self.ctx_.y[:upto], exog=self.ctx_.exs[:upto], order=self.order,
                       seasonal_order=self.seasonal_order, enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=0)

    def fit(self, ctx):
        """Fit the model once on the training history (up to the first test day)."""
        self.ctx_ = ctx
        self.res_ = self._fit_full(ctx.t0)
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t from the current fitted state and that day's exogenous row."""
        check_is_fitted(self, "res_")
        return float(self.res_.forecast(1, exog=self.ctx_.exs[t:t + 1])[0])

    def observe(self, t: int) -> None:
        """Append day t's actual and exogenous row to the fitted state (no refit), and re-estimate the coefficients every reestimate_every days."""
        check_is_fitted(self, "res_")
        ctx = self.ctx_
        i = t - ctx.t0
        self.res_ = self.res_.append(ctx.y[t:t + 1], exog=ctx.exs[t:t + 1], refit=False)
        if (i + 1) % self.reestimate_every == 0 and t + 1 < ctx.n:
            self.res_ = self._fit_full(t + 1)


class XGBoostForecaster(BaseForecaster):
    """Gradient-boosted trees on the feature matrix. Refits every refit_every days on an
    expanding window, then predicts from the actual features for the day."""
    name = "xgboost"

    def __init__(self, refit_every: int = 14, n_estimators: int = 300, max_depth: int = 3,
                 learning_rate: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, random_state: int = SEED):
        """Store the gradient-boosting hyperparameters and the refit cadence."""
        self.refit_every = refit_every
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, ctx):
        """Keep the context and clear the cached model so it refits on the first prediction."""
        self.ctx_ = ctx
        self.model_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days refit the booster on all data from warmup up to day t, then predict from day t's features."""
        check_is_fitted(self, "ctx_")
        ctx = self.ctx_
        i = t - ctx.t0
        if i % self.refit_every == 0:
            tr = slice(ctx.warmup, t)
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                random_state=self.random_state, objective="reg:squarederror")
            self.model_.fit(ctx.Xv[tr], ctx.y[tr])
        return float(self.model_.predict(ctx.Xv[t:t + 1])[0])


class ElasticNetForecaster(BaseForecaster):
    """Cross-validated ElasticNet on standardized features. Refits every refit_every days.
    Exposes the chosen alpha and l1_ratio from the most recent refit via .choice_."""
    name = "elasticnet"

    def __init__(self, refit_every: int = 14,
                 l1_ratio=(.1, .3, .5, .7, .9, .95, 1.0), cv=TimeSeriesSplit(n_splits=5),
                 max_iter: int = 5000, random_state: int = SEED):
        """Store the ElasticNet l1-ratio grid, the cross-validator, and the refit cadence. cv
        defaults to TimeSeriesSplit (time-ordered folds) because shuffling is invalid for time
        series; an int would mean shuffled KFold and is not used by default."""
        self.refit_every = refit_every
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, ctx):
        """Keep the context and clear the cached model, scaler, and chosen hyperparameters."""
        self.ctx_ = ctx
        self.model_ = None
        self.scaler_ = None
        self.choice_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days re-standardize the features and refit the cross-validated ElasticNet on data up to day t (recording the chosen alpha and l1_ratio), then predict from that day's scaled features."""
        check_is_fitted(self, "ctx_")
        ctx = self.ctx_
        i = t - ctx.t0
        if i % self.refit_every == 0:
            tr = slice(ctx.warmup, t)
            self.scaler_ = StandardScaler().fit(ctx.Xv[tr])
            self.model_ = ElasticNetCV(l1_ratio=list(self.l1_ratio), cv=self.cv,
                                       max_iter=self.max_iter,
                                       random_state=self.random_state).fit(
                                       self.scaler_.transform(ctx.Xv[tr]), ctx.y[tr])
            self.choice_ = {"alpha": float(self.model_.alpha_),
                            "l1_ratio": float(self.model_.l1_ratio_)}
        return float(self.model_.predict(self.scaler_.transform(ctx.Xv[t:t + 1]))[0])


class RandomForestForecaster(BaseForecaster):
    """Bagged regression trees on the feature matrix. Refits every refit_every days on an
    expanding window. Bagging (averaging many independent deep trees) produces a different error
    pattern than XGBoost's sequential boosting, so the two disagree in useful ways."""
    name = "random_forest"

    def __init__(self, refit_every: int = 14, n_estimators: int = 300, max_depth=None,
                 max_features=1.0, min_samples_leaf: int = 1, random_state: int = SEED):
        """Store the random-forest hyperparameters and the refit cadence."""
        self.refit_every = refit_every
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, ctx):
        """Keep the context and clear the cached model so it refits on the first prediction."""
        self.ctx_ = ctx
        self.model_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days refit the forest on all data from warmup up to day t, then predict from that day's features."""
        check_is_fitted(self, "ctx_")
        ctx = self.ctx_
        i = t - ctx.t0
        if i % self.refit_every == 0:
            tr = slice(ctx.warmup, t)
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state, n_jobs=-1)
            self.model_.fit(ctx.Xv[tr], ctx.y[tr])
        return float(self.model_.predict(ctx.Xv[t:t + 1])[0])


class ThetaForecaster(BaseForecaster):
    """The Theta method: split the series into a long-run trend line and a smoothed short-run
    part, forecast each, then add the weekly season back. A strong, cheap statistical model. It
    is univariate (uses only the demand history, not the driver columns) and re-fits each step on
    the history up to that day, which is cheap."""
    name = "theta"

    def __init__(self, season_length: int = 7, deseasonalize: bool = True, method: str = "auto"):
        """Store the seasonal period and the Theta-method options."""
        self.season_length = season_length
        self.deseasonalize = deseasonalize
        self.method = method

    def predict_step(self, t: int) -> float:
        """Refit the Theta model on the demand history up to day t and return its one-step forecast."""
        check_is_fitted(self, "ctx_")
        res = ThetaModel(self.ctx_.y[:t], period=self.season_length,
                         deseasonalize=self.deseasonalize, method=self.method).fit()
        return float(np.asarray(res.forecast(1))[0])


class LightGBMForecaster(BaseForecaster):
    """Light gradient boosting on the feature matrix. Same job as XGBoost but a different boosting
    library (leaf-wise tree growth), often faster and sometimes more accurate. A standard member of
    the model stack; the lightgbm package is required."""
    name = "lightgbm"

    def __init__(self, refit_every: int = 14, n_estimators: int = 300, num_leaves: int = 31,
                 learning_rate: float = 0.05, subsample: float = 0.8, colsample_bytree: float = 0.8,
                 min_child_samples: int = 20, reg_lambda: float = 1.0, random_state: int = SEED):
        """Store the LightGBM hyperparameters and the refit cadence."""
        self.refit_every = refit_every
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, ctx):
        """Keep the context and clear the cached model so it refits on the first prediction."""
        self.ctx_ = ctx
        self.model_ = None
        return self

    def predict_step(self, t: int) -> float:
        """Forecast day t. Every refit_every days refit the LightGBM model on all data from warmup up to day t, then predict from that day's features."""
        check_is_fitted(self, "ctx_")
        ctx = self.ctx_
        i = t - ctx.t0
        if i % self.refit_every == 0:
            tr = slice(ctx.warmup, t)
            self.model_ = lgb.LGBMRegressor(
                n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, subsample=self.subsample, subsample_freq=1,
                colsample_bytree=self.colsample_bytree, min_child_samples=self.min_child_samples,
                reg_lambda=self.reg_lambda, random_state=self.random_state, n_jobs=-1, verbosity=-1)
            # LightGBM 4.x auto-generates internal feature names (Column_0, ...) even when fitted
            # on a NumPy array; with scikit-learn 1.6+ this triggers a misleading predict-time
            # "X does not have valid feature names" warning even though fit and predict both use
            # the same nameless NumPy matrix. It is a known false positive (LightGBM #6798/#6934/
            # #7075) and does not affect results. Silence ONLY that exact message, scoped here, so
            # any other genuine warning still surfaces.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                    message="X does not have valid feature names",
                    category=UserWarning)
                self.model_.fit(ctx.Xv[tr], ctx.y[tr])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                message="X does not have valid feature names",
                category=UserWarning)
            return float(self.model_.predict(ctx.Xv[t:t + 1])[0])


def backtest_model(model: BaseForecaster, ctx: BacktestContext,
                   suppress_warnings: bool = False) -> np.ndarray:
    """Walk time forward one day at a time, collecting one-step-ahead forecasts. Each model
    handles its own refit cadence inside predict_step and observe.

    suppress_warnings silences only the statistical models' expected convergence and
    frequency warnings, so genuine warnings still surface."""
    def _run():
        """Run the rolling-origin loop: fit once, then for each test day record the one-step forecast and feed the actual back to the model."""
        model.fit(ctx)
        out = []
        for t in range(ctx.t0, ctx.n):
            out.append(model.predict_step(t))
            model.observe(t)
        return np.asarray(out, dtype=float)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", ValueWarning)
            return _run()
    return _run()


def build_registry(refit_every: int, sarimax_reestimate: int, seed: int, m: int) -> dict:
    """Map a name to a configured model. Add a model (next item) or tune one (the item after)
    by editing a single entry here."""
    reg = {
        "seasonal_naive": SeasonalNaiveForecaster(season_length=m),
        "holt_winters":   HoltWintersForecaster(season_length=m, refit_every=refit_every),
        "sarimax":        SARIMAXForecaster(reestimate_every=sarimax_reestimate),
        "xgboost":        XGBoostForecaster(refit_every=refit_every, random_state=seed),
        "elasticnet":     ElasticNetForecaster(refit_every=refit_every, random_state=seed),
        "random_forest":  RandomForestForecaster(refit_every=refit_every, random_state=seed),
        "theta":          ThetaForecaster(season_length=m),
        "lightgbm":       LightGBMForecaster(refit_every=refit_every, random_state=seed),
    }
    return reg


# Models that emit noisy convergence/frequency warnings; those specific warnings are silenced.
_QUIET_MODELS = {"holt_winters", "sarimax", "theta"}
SARIMAX_EXOG_COLS = ["is_holiday", "scheduled_surgeries", "census_heme_onc_lag1", "cbc_low_platelet_lag1"]


def run_model_selection(clean_df, test_days=168, refit_every=14, sarimax_reestimate=28,
                        seed=SEED, verbose=True):
    """Main selector:
    fits the stack, scores each model with a rolling-origin one-step-ahead back-test (MASE),
    trims the simple average to the models that beat the seasonal-naive baseline, picks the
    winner, and returns the scores, forecasts, actuals, dates, winner, members, features, and table.
    """
    df = clean_df.reset_index(drop=True)
    y = df["demand_model"].values.astype(float)
    Xdf = build_features(df)
    Xv = Xdf.values
    n = len(y); m = 7; warmup = 14
    min_train = 35
    requested = int(test_days)
    test_days = max(7, min(requested, n - min_train, max(14, int(0.30 * n))))
    t0 = n - test_days
    if t0 < min_train:
        raise ValueError(
            f"Not enough history to fit the seasonal models reliably: the back-test would train "
            f"on only {t0} days, but at least {min_train} are required (about five weeks; the "
            f"weekly models need at least two full weeks to initialize). Provide at least "
            f"{min_train + 7} days of daily data; one to three years is recommended."
        )
    if verbose and test_days != requested:
        print(f"note: data has {n} days, so the rolling back-test uses a {test_days}-day window "
              f"(training on the first {t0} days).")
    Q = np.mean(np.abs(y[m:t0] - y[:t0 - m]))           # in-sample seasonal-naive error: the MASE scale
    actual = y[t0:]
    dates = df["date"].values[t0:]

    def mase(p):
        """Mean Absolute Scaled Error of forecast array p against the held-out actuals, scaled by the in-sample seasonal-naive error Q."""
        return float(np.mean(np.abs(actual - p)) / Q)

    # exogenous matrix for SARIMAX: standardized known drivers, lagged where observed same-day
    ex = Xdf[SARIMAX_EXOG_COLS].copy()
    ex["census_heme_onc_lag1"] = ex["census_heme_onc_lag1"].bfill()
    ex["cbc_low_platelet_lag1"] = ex["cbc_low_platelet_lag1"].bfill()
    exs = ((ex - ex.mean()) / ex.std()).values

    ctx = BacktestContext(y=y, Xv=Xv, exs=exs, warmup=warmup, m=m, t0=t0, n=n)
    registry = build_registry(refit_every, sarimax_reestimate, seed, m)

    fc = {}
    for name, model in registry.items():
        fc[name] = backtest_model(model, ctx, suppress_warnings=name in _QUIET_MODELS)
    chosen = registry["elasticnet"].choice_

    scores = {k: mase(v) for k, v in fc.items()}
    base = list(registry.keys())          # every registered model flows into the table and ensemble
    members = [k for k in base if k != "seasonal_naive" and scores[k] < scores["seasonal_naive"]] or ["seasonal_naive"]
    fc["avg_trimmed"] = np.mean([fc[k] for k in members], axis=0)
    scores["avg_trimmed"] = mase(fc["avg_trimmed"])
    winner = min(scores, key=scores.get)

    table = pd.DataFrame({
        "MASE": pd.Series(scores),
        "MAE":  pd.Series({k: float(np.mean(np.abs(actual - v))) for k, v in fc.items()})
    }).loc[base + ["avg_trimmed"]].round({"MASE": 3, "MAE": 2})

    if verbose:
        print("Rolling-origin back-test over the last", test_days, "days, 1-step-ahead. Lower MASE is better.")
        print("trimmed average combines:", members)
        print("WINNER:", winner, "with MASE", round(scores[winner], 3))
        print("ElasticNet cross-validation chose alpha", round(chosen["alpha"], 4),
              "and l1_ratio", chosen["l1_ratio"], "(near 1 means mostly Lasso with a little Ridge)")
    return {"scores": scores, "forecasts": fc, "actual": actual, "dates": dates,
            "winner": winner, "members": members, "feature_cols": Xdf.columns.tolist(),
            "test_days": test_days, "elasticnet_choice": chosen, "table": table}
