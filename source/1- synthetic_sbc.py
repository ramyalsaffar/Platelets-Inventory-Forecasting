"""
Synthetic Stanford Blood Center (SBC) platelet dataset generator.

This is fake data designed to mimic SBC platelet operations. Nothing here reads real data; it is
the single data source for the whole pipeline (cleaning, feature building, model selection, tuning).

DESIGN GOALS
  - Every assumption is an explicit, documented parameter on SBCDataConfig (no hidden magic numbers).
  - With request_mode="true_demand" and n_days=1096 it reproduces the exact dataset produced by the legacy
    generator at the bottom of this file, so that baseline stays verifiable; the realistic default
    differs on purpose.
  - It produces the 20-column daily schema the cleaner and feature builder expect.

REQUEST MODEL (realistic by default):
  By default (request_mode="noisy"), observed `platelets_requested` is a noisy, optionally biased
  observation of the latent true demand, and units issued (`platelets_used`) are that order filled
  up to available stock, so used <= requested. De-censoring therefore recovers the orders, not the
  hidden clinical need, which is the realistic case. Set request_mode="true_demand" to make orders
  equal true demand; with that mode and n_days=1096 the generator reproduces the legacy
  dataset exactly.

WHAT IS MODELLED
  - Calendar: day of week, weekends, US federal holidays, day of year.
  - A latent autoregressive "hospital busyness" signal so lagged features stay informative.
  - Annual (winter-high) seasonality.
  - Hospital census (heme/onc, ICU, med/surg) and abnormal CBC counts driven by that population.
  - True platelet demand from calendar, drivers, noise, and rare surge days.
  - A baseline collection plan that slightly over-collects (so it wastes about ten percent, matching
    the Guan 2017 baseline) and ignores day of week, plus multi-day collection outages that cause
    occasional shortages.
  - Inventory simulated day by day with five-day shelf life, FIFO issue, and end-of-day stock by age.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# Column order the data dictionary and the cleaner expect. The per-age stock columns (one per day
# of remaining shelf life) are generated FROM the shelf life, so the schema scales to 5- or 7-day
# platelets. SCHEMA_COLUMNS stays the canonical 5-day list, so any code that imports it is
# unchanged; for any other shelf life call schema_columns(shelf_life_days).
def stock_columns(shelf_life_days: int = 5) -> list:
    """End-of-day stock column names, one per day of remaining shelf life (oldest first)."""
    return [f"stock_{k}day_left" for k in range(1, shelf_life_days + 1)]


def schema_columns(shelf_life_days: int = 5) -> list:
    """Full daily column order for a given shelf life. With shelf_life_days=5 this returns the exact
    canonical 20-column list that the cleaner and data dictionary expect."""
    return (["date", "platelets_used", "platelets_requested", "platelets_received", "platelets_expired"]
            + stock_columns(shelf_life_days)
            + ["day_of_week", "is_weekend", "is_holiday",
               "census_heme_onc", "census_icu", "census_med_surg", "scheduled_surgeries",
               "cbc_low_platelet", "cbc_abnormal_mcv", "cbc_high_rdw"])


SCHEMA_COLUMNS = schema_columns(5)   # canonical 5-day schema (the default the cleaner expects)


@dataclass
class SBCDataConfig:
    """All generator assumptions. Defaults reproduce the project's dataset exactly."""
    # --- size and reproducibility ---
    start_date: str = "2022-01-01"
    n_days: int = 1826                      # five years from 2022-01-01 (see sizing notes)
    seed: int = 42

    # --- latent busyness and annual seasonality ---
    ar_coef: float = 0.85                   # persistence of the latent busyness signal (AR1)
    season_phase_doy: int = 20              # day of year where the annual cycle peaks (mid-January)
    season_period: int = 365

    # --- hospital census (level + busyness loading + seasonal loading + noise sigma + floor) ---
    heme_onc: tuple = (45.0, 5.0, 4.0, 3.0, 5)
    icu: tuple = (70.0, 6.0, 3.0, 5.0, 10)
    med_surg: tuple = (300.0, 25.0, 10.0, 15.0, 50)
    # scheduled surgeries: weekday level, weekend level, holiday reduction, noise sigma
    surg_weekday: float = 60.0
    surg_weekend: float = 4.0
    surg_holiday_drop: float = 0.7
    surg_sigma: float = 6.0

    # --- abnormal CBC counts (coefficient on heme/onc, extra term, noise sigma) ---
    cbc_low: tuple = (0.6, 0.0, 4.0)        # low platelet count
    cbc_mcv: tuple = (0.8, 0.0, 6.0)        # abnormal mean corpuscular volume
    cbc_rdw: tuple = (1.0, 0.1, 8.0)        # high red cell distribution width (extra term on med/surg)

    # --- true platelet demand ---
    demand_base: float = 35.6
    dow_offset: tuple = (2, 4, 4, 3, 1, -8, -10)   # Mon..Sun; weekdays high, weekends low
    demand_season_amp: float = 3.0
    demand_coef_heme: float = 0.06
    demand_coef_cbc_low: float = 0.04
    demand_coef_surg: float = 0.04
    demand_sigma: float = 4.0
    demand_holiday_drop: float = 0.35
    moderate_surge_prob: float = 0.008
    moderate_surge_range: tuple = (15, 32)
    big_surge_prob: float = 0.003
    big_surge_range: tuple = (55, 90)

    # --- collection outages (storms / closures) ---
    n_outages: int = 12
    outage_dur_range: tuple = (3, 7)
    outage_mult_range: tuple = (0.15, 0.40)

    # --- baseline collection plan and inventory ---
    shelf_life_days: int = 5                # 5-day US default; set to 7 for screened / pathogen-reduced units
    capacity_days: float = 6.0              # cap on hand = capacity_days * plan_capacity_unit
    plan_capacity_unit: float = 35.0
    plan_weekday: float = 38.5
    plan_weekend: float = 26.0
    plan_holiday_drop: float = 0.6
    collect_factor: float = 1.18            # >1 means slight over-collection (creates ~10% waste)
    collect_sigma: float = 4.0

    # --- optional realism: observed requested orders vs true demand ---
    request_mode: str = "noisy"             # "noisy" (realistic default) or "true_demand" (reproduces project data)
    request_bias: float = 0.0               # systematic over/under-ordering when request_mode="noisy"
    request_sigma: float = 3.0              # observation noise on requested when request_mode="noisy"


def generate_sbc_platelet_data(config: SBCDataConfig = None) -> pd.DataFrame:
    """Generate the synthetic daily platelet table. Returns a DataFrame with SCHEMA_COLUMNS. With
    request_mode="true_demand" and n_days=1096 this reproduces the project's original dataset exactly;
    the realistic default (noisy orders, five years) differs on purpose."""
    c = config or SBCDataConfig()
    rng = np.random.default_rng(c.seed)

    dates = pd.date_range(start=c.start_date, periods=c.n_days, freq="D")
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values
    hol = USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max())
    is_holiday = pd.Series(dates).isin(hol).astype(int).values
    is_weekend = (dow >= 5).astype(int)

    # Latent autoregressive busyness (draw order preserved for exact reproduction).
    z = np.zeros(c.n_days)
    for t in range(1, c.n_days):
        z[t] = c.ar_coef * z[t - 1] + rng.normal(0, 1)
    season = np.cos(2 * np.pi * (doy - c.season_phase_doy) / c.season_period)

    def census(p):
        """Draw one hospital-census series from its parameter tuple (level, busyness loading, seasonal loading, noise sigma, floor)."""
        level, zload, sload, sigma, floor = p
        return np.round(level + zload * z + sload * season + rng.normal(0, sigma, c.n_days)).clip(min=floor)

    census_heme_onc = census(c.heme_onc)
    census_icu = census(c.icu)
    census_med_surg = census(c.med_surg)
    scheduled_surgeries = np.round(
        np.where(is_weekend == 1, c.surg_weekend, c.surg_weekday) * (1 - c.surg_holiday_drop * is_holiday)
        + rng.normal(0, c.surg_sigma, c.n_days)).clip(min=0)

    cbc_low_platelet = np.round(c.cbc_low[0] * census_heme_onc + c.cbc_low[1] * census_med_surg
                                + rng.normal(0, c.cbc_low[2], c.n_days)).clip(min=0)
    cbc_abnormal_mcv = np.round(c.cbc_mcv[0] * census_heme_onc + c.cbc_mcv[1] * census_med_surg
                                + rng.normal(0, c.cbc_mcv[2], c.n_days)).clip(min=0)
    cbc_high_rdw = np.round(c.cbc_rdw[0] * census_heme_onc + c.cbc_rdw[1] * census_med_surg
                            + rng.normal(0, c.cbc_rdw[2], c.n_days)).clip(min=0)

    dow_offset = np.array(c.dow_offset)[dow]
    demand = (c.demand_base + dow_offset + c.demand_season_amp * season
              + c.demand_coef_heme * (census_heme_onc - census_heme_onc.mean())
              + c.demand_coef_cbc_low * (cbc_low_platelet - cbc_low_platelet.mean())
              + c.demand_coef_surg * (scheduled_surgeries - scheduled_surgeries.mean())
              + rng.normal(0, c.demand_sigma, c.n_days))
    demand = demand * (1 - c.demand_holiday_drop * is_holiday)
    moderate = rng.random(c.n_days) < c.moderate_surge_prob
    demand[moderate] += rng.uniform(*c.moderate_surge_range, moderate.sum())
    big = rng.random(c.n_days) < c.big_surge_prob
    demand[big] += rng.uniform(*c.big_surge_range, big.sum())
    true_demand = np.round(demand).clip(min=0).astype(int)

    # Observed order signal. In the realistic default ("noisy"), orders are a noisy, optionally biased
    # view of the latent true demand; units are then issued against the ORDER (used <= requested), so
    # de-censoring recovers the orders rather than the hidden clinical need. In "true_demand" mode the
    # order equals true demand, which reproduces the project's original data.
    if c.request_mode == "noisy":
        rng2 = np.random.default_rng(c.seed + 1)            # separate stream; keeps "true_demand" exact
        requested_signal = np.round(true_demand * (1 + c.request_bias)
                                    + rng2.normal(0, c.request_sigma, c.n_days)).clip(min=0).astype(int)
    elif c.request_mode == "true_demand":
        requested_signal = true_demand.copy()
    else:
        raise ValueError("request_mode must be 'true_demand' or 'noisy'")

    # Multi-day collection outages.
    disrupt_mult = np.ones(c.n_days)
    for _ in range(c.n_outages):
        s = rng.integers(10, c.n_days - 7)
        dur = rng.integers(*c.outage_dur_range)
        disrupt_mult[s:s + dur] = rng.uniform(*c.outage_mult_range)

    # Inventory simulation under the baseline plan.
    shelf = c.shelf_life_days
    cap = c.capacity_days * c.plan_capacity_unit
    plan = np.where(is_weekend == 1, c.plan_weekend, c.plan_weekday) * (1 - c.plan_holiday_drop * is_holiday)
    stock = [0] * shelf
    received, used, requested, expired = [], [], [], []
    stock_age = [[] for _ in range(shelf)]
    for t in range(c.n_days):
        exp = stock[0]                                  # 1-day-left units expire overnight
        for k in range(shelf - 1):
            stock[k] = stock[k + 1]
        stock[shelf - 1] = 0
        on_hand = sum(stock)
        order = 0 if on_hand > cap else int(max(c.collect_factor * plan[t] + rng.normal(0, c.collect_sigma), 0))
        order = int(order * disrupt_mult[t])
        stock[shelf - 1] += order
        d = int(requested_signal[t]); to_issue = d; served = 0   # issue against the order; used <= requested
        for k in range(shelf):                          # FIFO: issue oldest first
            take = min(stock[k], to_issue)
            stock[k] -= take; served += take; to_issue -= take
            if to_issue == 0:
                break
        for k in range(shelf):
            stock_age[k].append(stock[k])
        received.append(order); used.append(served); requested.append(d); expired.append(exp)

    used = np.array(used); requested = np.array(requested)

    data = {
        "date": dates,
        "platelets_used": used, "platelets_requested": requested,
        "platelets_received": received, "platelets_expired": expired,
        "day_of_week": dates.day_name(), "is_weekend": is_weekend, "is_holiday": is_holiday,
        "census_heme_onc": census_heme_onc.astype(int), "census_icu": census_icu.astype(int),
        "census_med_surg": census_med_surg.astype(int), "scheduled_surgeries": scheduled_surgeries.astype(int),
        "cbc_low_platelet": cbc_low_platelet.astype(int), "cbc_abnormal_mcv": cbc_abnormal_mcv.astype(int),
        "cbc_high_rdw": cbc_high_rdw.astype(int),
    }
    for k, col_name in enumerate(stock_columns(shelf)):   # one stock column per day of shelf life
        data[col_name] = stock_age[k]
    df = pd.DataFrame(data)
    return df[schema_columns(shelf)]                       # canonical column order for this shelf life


def summarize_realism(df: pd.DataFrame) -> dict:
    """Report the properties a defensible synthetic dataset should have, so the realism is explicit."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    dow = d["date"].dt.dayofweek
    month = d["date"].dt.month
    stock_cols = [col for col in d.columns if col.startswith("stock_") and col.endswith("day_left")]
    total_stock = d[stock_cols].sum(axis=1)
    flow = d["platelets_received"] - d["platelets_used"] - d["platelets_expired"]
    identity_ok = bool(np.allclose(total_stock.diff().iloc[1:], flow.iloc[1:], atol=1e-9))
    shortage_days = int((d["platelets_requested"] > d["platelets_used"]).sum())
    requested_equals_used_plus = (d["platelets_requested"] >= d["platelets_used"]).mean()
    return {
        "rows": len(d),
        "weekday_mean_used": round(float(d.loc[dow < 5, "platelets_used"].mean()), 2),
        "weekend_mean_used": round(float(d.loc[dow >= 5, "platelets_used"].mean()), 2),
        "winter_mean_used_JanFeb": round(float(d.loc[month.isin([1, 2]), "platelets_used"].mean()), 2),
        "summer_mean_used_JulAug": round(float(d.loc[month.isin([7, 8]), "platelets_used"].mean()), 2),
        "holiday_mean_used": round(float(d.loc[d["is_holiday"] == 1, "platelets_used"].mean()), 2),
        "nonholiday_mean_used": round(float(d.loc[d["is_holiday"] == 0, "platelets_used"].mean()), 2),
        "shortage_days": shortage_days,
        "waste_pct": round(100 * d["platelets_expired"].sum() / max(d["platelets_received"].sum(), 1), 2),
        "stock_never_negative": bool((total_stock >= 0).all()),
        "inventory_identity_holds": identity_ok,
        "requested_geq_used_fraction": round(float(requested_equals_used_plus), 3),
    }

# ----------------------------------------------------------------------------
# LEGACY generator (kept for reference only; not used by the pipeline).
# generate_platelet_data is the project's ORIGINAL data generator.
# generate_sbc_platelet_data above supersedes it: with request_mode="true_demand" and
# n_days=1096 the new generator reproduces this one's output byte for byte. It is kept so the
# original baseline stays runnable for anyone reproducing the early results.
# ----------------------------------------------------------------------------
def generate_platelet_data(start_date="2022-01-01", n_days=1096, seed=42):
    """Create a realistic fake daily platelet dataset for SBC (platelets only).

    Demand is built from calendar + clinical drivers + noise + rare surge days.
    Inventory is simulated under a simple current (baseline) collection plan that slightly
    over-collects, so it wastes ~10% (the Guan 2017 baseline). Multi-day collection outages
    create occasional shortages. Returns a tidy daily DataFrame with the 20 dictionary columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.values          # 0=Mon ... 6=Sun
    doy = dates.dayofyear.values
    hol = USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max())
    is_holiday = pd.Series(dates).isin(hol).astype(int).values
    is_weekend = (dow >= 5).astype(int)

    # Latent hospital "busyness" with persistence (AR1) so lagged features stay informative.
    z = np.zeros(n_days)
    for t in range(1, n_days):
        z[t] = 0.85 * z[t-1] + rng.normal(0, 1)
    season = np.cos(2 * np.pi * (doy - 20) / 365)   # winter (flu season) high, summer low

    # Hospital census. Heme/onc is the main platelet driver.
    census_heme_onc = np.round(45 + 5*z + 4*season + rng.normal(0, 3, n_days)).clip(min=5)
    census_icu      = np.round(70 + 6*z + 3*season + rng.normal(0, 5, n_days)).clip(min=10)
    census_med_surg = np.round(300 + 25*z + 10*season + rng.normal(0, 15, n_days)).clip(min=50)
    scheduled_surgeries = np.round(np.where(is_weekend == 1, 4, 60) * (1 - 0.7*is_holiday)
                                   + rng.normal(0, 6, n_days)).clip(min=0)

    # Abnormal CBC counts, driven by the heme/onc population.
    cbc_low_platelet = np.round(0.6*census_heme_onc + rng.normal(0, 4, n_days)).clip(min=0)
    cbc_abnormal_mcv = np.round(0.8*census_heme_onc + rng.normal(0, 6, n_days)).clip(min=0)
    cbc_high_rdw     = np.round(1.0*census_heme_onc + 0.1*census_med_surg + rng.normal(0, 8, n_days)).clip(min=0)

    # TRUE platelet demand (clinical need) = calendar + drivers + noise + surges.
    dow_offset = np.array([2, 4, 4, 3, 1, -8, -10])[dow]      # weekdays high, weekends low
    demand = (35.6 + dow_offset + 3.0*season
              + 0.06*(census_heme_onc - census_heme_onc.mean())
              + 0.04*(cbc_low_platelet - cbc_low_platelet.mean())
              + 0.04*(scheduled_surgeries - scheduled_surgeries.mean())
              + rng.normal(0, 4.0, n_days))
    demand = demand * (1 - 0.35*is_holiday)                   # fewer transfusions on holidays
    moderate = rng.random(n_days) < 0.008
    demand[moderate] += rng.uniform(15, 32, moderate.sum())   # busy surges
    big = rng.random(n_days) < 0.003
    demand[big] += rng.uniform(55, 90, big.sum())             # rare mass-casualty
    true_demand = np.round(demand).clip(min=0).astype(int)

    # Multi-day collection outages (storm / holiday closure) -> occasional shortages.
    disrupt_mult = np.ones(n_days)
    for _ in range(12):
        s = rng.integers(10, n_days - 7); dur = rng.integers(3, 7)
        disrupt_mult[s:s+dur] = rng.uniform(0.15, 0.40)

    # Baseline current collection plan: slightly over-collects (waste), ignores day-of-week.
    SHELF = 5; cap = 6 * 35
    plan = np.where(is_weekend == 1, 26.0, 38.5) * (1 - 0.6*is_holiday)
    COLLECT_FACTOR = 1.18
    stock = [0]*SHELF                       # stock[k] = units with (k+1) days left
    received, used, requested, expired = [], [], [], []
    stock_age = [[] for _ in range(SHELF)]
    for t in range(n_days):
        # Start of day: units that aged out overnight expire (the 1-day-left units from last night).
        exp = stock[0]
        for k in range(SHELF-1):
            stock[k] = stock[k+1]
        stock[4] = 0
        # Collect fresh units (5 days left), reduced during an outage.
        on_hand = sum(stock)
        order = 0 if on_hand > cap else int(max(COLLECT_FACTOR*plan[t] + rng.normal(0, 4), 0))
        order = int(order * disrupt_mult[t])
        stock[4] += order
        # Issue oldest first (FIFO) to meet demand.
        d = int(true_demand[t]); to_issue = d; served = 0
        for k in range(SHELF):
            take = min(stock[k], to_issue)
            stock[k] -= take; served += take; to_issue -= take
            if to_issue == 0:
                break
        for k in range(SHELF):
            stock_age[k].append(stock[k])   # end-of-day stock by age
        received.append(order); used.append(served); requested.append(d); expired.append(exp)

    df = pd.DataFrame({
        "date": dates,
        "platelets_used": used, "platelets_requested": requested,
        "platelets_received": received, "platelets_expired": expired,
        "stock_1day_left": stock_age[0], "stock_2day_left": stock_age[1], "stock_3day_left": stock_age[2],
        "stock_4day_left": stock_age[3], "stock_5day_left": stock_age[4],
        "day_of_week": dates.day_name(), "is_weekend": is_weekend, "is_holiday": is_holiday,
        "census_heme_onc": census_heme_onc.astype(int), "census_icu": census_icu.astype(int),
        "census_med_surg": census_med_surg.astype(int), "scheduled_surgeries": scheduled_surgeries.astype(int),
        "cbc_low_platelet": cbc_low_platelet.astype(int), "cbc_abnormal_mcv": cbc_abnormal_mcv.astype(int),
        "cbc_high_rdw": cbc_high_rdw.astype(int),
    })
    return df
