"""
cleaning.py
===========
Standalone data-cleaning module for the platelet inventory pipeline.

It is schema-aware: each column's role (from the data dictionary) decides how it is treated.
It contains:
  - DATA_DICTIONARY : the column -> role mapping the cleaner uses
  - clean_platelet_data(df, ...) : the cleaning function

Repairs are conservative: anything ambiguous is flagged for review rather than altered.

Acronyms used in the column descriptions:
  CBC = Complete Blood Count (a routine blood test)
  ICU = Intensive Care Unit
  MCV = Mean Corpuscular Volume (the average red blood cell size)
  RDW = Red cell Distribution Width (how much red blood cell size varies)

Usage:
  from cleaning import clean_platelet_data
  clean_df, report = clean_platelet_data(raw_df)
"""

import re
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# ----------------------------------------------------------------------------
# Data dictionary: every column, its meaning, units, and role.
# Columns mirror Guan et al. 2017 (PNAS), simplified for a platelets-only daily model.
# Each column's "role" is what the cleaner uses to decide how to treat that column.
# ----------------------------------------------------------------------------
_rows = [
    ("date",                "Calendar day (one row per day).",                                              "date",              "index"),
    ("platelets_used",      "Platelet units transfused that day. This is the value we predict.",            "units (count)",     "target"),
    ("platelets_requested", "Platelet units the hospital requested that day. On shortage days this is higher than used, so we use it to avoid under-counting demand.", "units (count)", "demand"),
    ("platelets_received",  "Fresh platelet units that became usable that day (after testing).",            "units (count)",     "supply"),
    ("platelets_expired",   "Platelet units that expired (outdated) that day.",                             "units (count)",     "waste"),
    ("stock_1day_left",     "Units in stock with 1 day left before expiry (use first).",                    "units (count)",     "inventory by age"),
    ("stock_2day_left",     "Units in stock with 2 days left before expiry.",                               "units (count)",     "inventory by age"),
    ("stock_3day_left",     "Units in stock with 3 days left before expiry.",                               "units (count)",     "inventory by age"),
    ("stock_4day_left",     "Units in stock with 4 days left before expiry.",                               "units (count)",     "inventory by age"),
    ("stock_5day_left",     "Units in stock with 5 days left before expiry (freshest).",                    "units (count)",     "inventory by age"),
    ("day_of_week",         "Day name, Monday to Sunday.",                                                  "text",              "calendar"),
    ("is_weekend",          "1 if Saturday or Sunday, else 0.",                                             "flag (0/1)",        "calendar"),
    ("is_holiday",          "1 if a US federal holiday, else 0.",                                           "flag (0/1)",        "calendar"),
    ("census_heme_onc",     "Hematology and oncology inpatients (the heaviest platelet users).",            "patients (count)",  "hospital census"),
    ("census_icu",          "Intensive care unit inpatients.",                                              "patients (count)",  "hospital census"),
    ("census_med_surg",     "General medicine and surgery inpatients.",                                     "patients (count)",  "hospital census"),
    ("scheduled_surgeries", "Surgeries planned for the next day.",                                          "procedures (count)","hospital census"),
    ("cbc_low_platelet",    "Patients with a low platelet count that day (strongest clinical driver of need).", "patients (count)","abnormal CBC"),
    ("cbc_abnormal_mcv",    "Patients with an abnormal MCV (mean corpuscular volume, the average red blood cell size) that day. Flags chemo and marrow-suppressed patients, who are heavy platelet users.", "patients (count)","abnormal CBC"),
    ("cbc_high_rdw",        "Patients with a high RDW (red cell distribution width, how much red cell size varies) that day. Also flags chemo and marrow-suppressed patients.", "patients (count)","abnormal CBC"),
]
DATA_DICTIONARY = pd.DataFrame(_rows, columns=["column", "meaning", "unit_or_type", "role"])


def _nan_run_length(mask):
    """For each missing (True) position in a boolean Series, the length of its consecutive run of
    missing values, and 0 where present. Used so we fill ONLY gaps no longer than max_gap_fill."""
    grp = (mask != mask.shift()).cumsum()
    run = mask.astype(int).groupby(grp).transform("sum")
    return run.where(mask, 0).astype(int)


def _series_diagnostics(s, period=7):
    """Optional report-only diagnostics on the demand series: ADF stationarity test and STL
    seasonal/trend strength. ADF = Augmented Dickey-Fuller (a test for whether a series is stable
    over time). STL = Seasonal-Trend decomposition using Loess. statsmodels is imported lazily, so
    the cleaner still runs without it; missing pieces are reported as None."""
    out = {}
    s = pd.Series(s).dropna()
    try:
        from statsmodels.tsa.stattools import adfuller
        p = float(adfuller(s.values, autolag="AIC")[1])
        out["adf_pvalue"] = round(p, 4); out["is_stationary_adf_5pct"] = bool(p < 0.05)
    except Exception:
        out["adf_pvalue"] = None; out["is_stationary_adf_5pct"] = None
    try:
        from statsmodels.tsa.seasonal import STL
        if len(s) >= 2 * period:
            r = STL(s.values, period=period, robust=True).fit()
            vr = float(np.var(r.resid))
            vs = float(np.var(r.seasonal + r.resid)); vt = float(np.var(r.trend + r.resid))
            out["seasonal_strength"] = round(max(0.0, 1 - vr / vs), 3) if vs else None
            out["trend_strength"]    = round(max(0.0, 1 - vr / vt), 3) if vt else None
        else:
            out["seasonal_strength"] = None; out["trend_strength"] = None
    except Exception:
        out["seasonal_strength"] = None; out["trend_strength"] = None
    return out


def clean_platelet_data(df, roles=None, max_gap_fill=2, outlier_k=4.0, recon_tol=1,
                        high_mult=5.0, diagnostics=True, verbose=True):
    """Clean a daily platelet table. Repairs only what is unambiguous and FLAGS what needs
    human judgment. Schema-aware: each column's role (from the data dictionary) decides its rule.
    Adds: demand_model (de-censored target), is_shortage, row_was_imputed, inventory_mismatch,
    is_demand_outlier, has_high_outlier, usage_avg_7d (leakage-safe). Tolerant of missing or
    extra columns. Returns (clean_df, report).

    NOTE on demand_model: it is a LOWER BOUND on true clinical demand. max(used, requested)
    recovers the order; if the order was itself trimmed because the desk knew stock was low, real
    demand is higher. A heavier censored-demand model (Tobit, expectation-maximization) is the
    upgrade path, but is unnecessary while the request column is recorded."""
    if roles is None:
        roles = dict(zip(DATA_DICTIONARY["column"], DATA_DICTIONARY["role"]))
    else:
        roles = dict(roles)                       # copy so we never mutate the caller's dict

    # Recognize stock-by-age columns by NAME PATTERN (stock_<N>day_left), not by the fixed 5-row
    # data dictionary. This makes a 5-day, 7-day, or any-shelf-life inventory behave identically.
    # Without it the cleaner ignores stock_6day_left / stock_7day_left, so they are never coerced
    # or gap-filled and, worse, the inventory reconciliation in step 5 sums only the first 5 stock
    # columns, undercounts total stock, and wrongly flags most days as mismatches.
    _stock_pat = re.compile(r"^stock_\d+day_left$")
    for c in df.columns:
        if _stock_pat.match(str(c)):
            roles[c] = "inventory by age"

    df = df.copy(); rep = {}
    count_roles = {"target", "demand", "supply", "waste", "inventory by age", "hospital census", "abnormal CBC"}
    num_cols = [c for c in df.columns if roles.get(c) in count_roles]

    # 1) date -> datetime, drop unparseable, drop duplicate dates, sort, reindex to a full daily range
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rep["rows_dropped_bad_date"] = int(df["date"].isna().sum())
    df = df[df["date"].notna()]
    dup_mask = df["date"].duplicated(keep=False)
    rep["conflicting_duplicate_dates"] = int(sum(
        g.drop_duplicates().shape[0] > 1 for _, g in df[dup_mask].groupby("date"))) if dup_mask.any() else 0
    rep["duplicate_dates_removed"] = int(df["date"].duplicated().sum())
    df = df.sort_values("date").drop_duplicates("date")
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    rep["missing_calendar_days_inserted"] = int(len(full) - df["date"].nunique())
    df = df.set_index("date").reindex(full); df.index.name = "date"; df = df.reset_index()

    # 2) numeric columns: coerce to numbers, treat impossible negatives as missing
    neg_total = 0
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_neg = int((df[c] < 0).sum()); neg_total += n_neg
        if n_neg:
            df.loc[df[c] < 0, c] = np.nan
    rep["negative_values_fixed"] = neg_total

    # 3) fill ONLY short gaps (a run of <= max_gap_fill consecutive missing days). A LONGER gap is
    #    left entirely for human review, never partially filled from its edges. The target columns
    #    (platelets_used, platelets_requested) are filled PAST-ONLY (forward fill) so an imputed day
    #    never borrows a future value; driver columns use two-sided interpolation, which is safe
    #    because the models only ever see those drivers lagged (shifted into the past).
    target_cols = [c for c in df.columns if roles.get(c) in {"target", "demand"}]
    imputed_row = pd.Series(False, index=df.index)
    for c in num_cols:
        miss = df[c].isna()
        if not miss.any():
            continue
        run_len = _nan_run_length(miss)
        short = miss & (run_len <= max_gap_fill)         # within the limit: fill; longer runs: leave
        if c in target_cols:
            candidate = df[c].ffill()                    # past-only carry-forward, no future leakage
        else:
            candidate = df[c].interpolate(method="linear", limit_direction="both")
        df[c] = df[c].where(~short, candidate.round())   # write fills only at short-run positions
        imputed_row |= short & df[c].notna()
    rep["rows_short_gap_imputed"] = int(imputed_row.sum())
    rep["cells_left_missing_long_gap"] = int(df[num_cols].isna().sum().sum())   # all cells still missing
    df["row_was_imputed"] = imputed_row.astype(int)

    # 3b) high-side plausibility: flag impossibly large values such as a mistyped
    #     999999. Scale is the column's 99th percentile (its true upper range), so legitimate spikes
    #     and surges are not flagged; only values many times beyond the observed range trip it.
    high_flag = pd.Series(False, index=df.index); high_cols = []
    for c in num_cols:
        scale = float(df[c].quantile(0.99))              # robust upper range of the column
        if scale <= 0:
            continue
        col_hi = (df[c] > high_mult * scale).fillna(False)
        if col_hi.any():
            high_flag |= col_hi; high_cols.append(c)
    df["has_high_outlier"] = high_flag.astype(int)
    rep["high_value_outliers_flagged"] = int(high_flag.sum())
    rep["high_value_outlier_columns"] = high_cols

    # 4) recompute calendar columns from the date (the source of truth); create them if absent,
    #    so you only need to supply the date and the cleaner fills day_of_week, is_weekend, is_holiday
    hol = USFederalHolidayCalendar().holidays(start=df["date"].min(), end=df["date"].max())
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"]  = df["date"].isin(hol).astype(int)

    # 5) inventory reconciliation: received - used - expired should equal the change in total stock
    stock_cols = [c for c in df.columns if roles.get(c) == "inventory by age"]
    if stock_cols and {"platelets_received", "platelets_used", "platelets_expired"} <= set(df.columns):
        chg = df[stock_cols].sum(axis=1).diff()
        flow = df["platelets_received"] - df["platelets_used"] - df["platelets_expired"]
        mismatch = (chg - flow).abs() > recon_tol; mismatch.iloc[0] = False   # tol=1 absorbs 1-unit rounding from imputation
        df["inventory_mismatch"] = mismatch.astype(int)
        rep["inventory_mismatch_rows"] = int(mismatch.sum())

    # 6) de-censor demand: on shortage days, units used understates need, so use units requested.
    #    NOTE: demand_model is a LOWER BOUND on true clinical demand (see the function docstring).
    if {"platelets_used", "platelets_requested"} <= set(df.columns):
        df["is_shortage"] = (df["platelets_requested"] > df["platelets_used"]).astype(int)
        df["demand_model"] = np.maximum(df["platelets_used"], df["platelets_requested"])
        rep["shortage_days_decensored"] = int(df["is_shortage"].sum())
    else:
        df["demand_model"] = df.get("platelets_used")
        rep["shortage_days_decensored"] = 0

    # 7) flag rare surge days on the de-censored demand (robust, relative to each weekday level).
    #    Real events are FLAGGED, never removed, so the model step can choose to down-weight them.
    s = df["demand_model"].astype(float)
    base = df.groupby(df["date"].dt.dayofweek)["demand_model"].transform("median")
    resid = s - base
    mad = (resid - resid.median()).abs().median()
    rz = 0.6745 * (resid - resid.median()) / (mad if mad else 1.0)
    df["is_demand_outlier"] = (rz.abs() > outlier_k).astype(int)
    rep["demand_outliers_flagged"] = int(df["is_demand_outlier"].sum())

    # 8) leakage-safe feature: average daily use over the PREVIOUS 7 days (shift(1) keeps today out)
    df["usage_avg_7d"] = df["demand_model"].shift(1).rolling(7, min_periods=1).mean().round(1)

    # 9) optional report-only diagnostics: stationarity (ADF) and STL seasonal
    #    and trend strength on the demand series. Never alters the data.
    if diagnostics:
        rep.update(_series_diagnostics(df["demand_model"], period=7))

    if verbose:
        print("CLEANING REPORT")
        for k, v in rep.items():
            print(f"  {k:34s}: {v}")
    return df, rep
