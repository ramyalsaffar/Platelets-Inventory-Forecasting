"""
Feature engineering for the platelet forecasting models.

Run order: after cleaning.py and before forecasters.py. build_features turns a cleaned daily
table (which must contain demand_model and usage_avg_7d) into the leakage-safe feature matrix the
models consume. forecasters.py and tuning.py both call build_features, so it must be defined first.
"""
import numpy as np
import pandas as pd


def build_features(df):
    """Build a leakage-safe feature table for the machine-learning models.
    Every column uses only information available before the day being predicted."""
    X = pd.DataFrame(index=df.index)
    d = df["demand_model"].astype(float)
    X["lag_1"]  = d.shift(1)                 # yesterday
    X["lag_7"]  = d.shift(7)                 # same weekday last week
    X["lag_14"] = d.shift(14)               # two weeks back
    X["roll7_mean"] = df["usage_avg_7d"]     # average use over the previous 7 days (already shifted)
    X["roll7_std"]  = d.shift(1).rolling(7).std()   # how bumpy the last week was
    dow = df["date"].dt.dayofweek
    for k in range(1, 7):
        X[f"dow_{k}"] = (dow == k).astype(int)      # day-of-week dummies (Monday is the reference)
    X["is_holiday"] = df["is_holiday"]              # known ahead
    doy = df["date"].dt.dayofyear
    X["sin_year"] = np.sin(2*np.pi*doy/365)         # season (smooth yearly cycle)
    X["cos_year"] = np.cos(2*np.pi*doy/365)
    X["scheduled_surgeries"] = df["scheduled_surgeries"]   # planned a day ahead, so known
    for c in ["census_heme_onc", "census_icu", "census_med_surg",
              "cbc_low_platelet", "cbc_abnormal_mcv", "cbc_high_rdw"]:
        X[f"{c}_lag1"] = df[c].shift(1)             # observed same-day -> use yesterday to avoid leakage
    return X