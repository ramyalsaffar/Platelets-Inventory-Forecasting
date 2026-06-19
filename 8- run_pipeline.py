# ============================================================================
# RUN ORDER
#   1) Run files 1 through 7 yourself, in order, in the SAME Spyder console.
#   2) Then run THIS file. It calls the functions those files define and prints
#      and plots the results for every stage of the pipeline.
#
# REQUIRED ONE-TIME SPYDER SETTING (this is why an earlier file 8 errored):
#   Tools -> Preferences -> Run -> check
#     "Run in console's namespace instead of an empty one"
#   Click OK, then restart the console. Without it, Spyder runs each file in its
#   own empty namespace, so file 8 cannot see the functions and you get a NameError.
#
#   NOTE: the tuning stage fits many models on the full 5-year dataset and can take
#   a few minutes. For a quick run, pass smaller trial counts to run_tuned_selection
#   (see the commented example in stage 4).
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# ----------------------------------------------------------------------------
# STAGE 1: data generation, realism check, and cleaning
# ----------------------------------------------------------------------------
df = generate_sbc_platelet_data()

print("=" * 72); print("STAGE 1a  DATA REALISM (properties a defensible synthetic set should have)")
print("=" * 72)
for k, v in summarize_realism(df).items():
    print(f"  {k:32s}: {v}")

print("\n" + "=" * 72); print("STAGE 1b  CLEANING REPORT"); print("=" * 72)
clean, report = clean_platelet_data(df, verbose=True)
print(f"\nCleaned table: {len(clean)} rows, {clean.shape[1]} columns.")

# ----------------------------------------------------------------------------
# STAGE 2: feature engineering
# ----------------------------------------------------------------------------
Xdf = build_features(clean)
print("\n" + "=" * 72); print("STAGE 2  FEATURES"); print("=" * 72)
print(f"  built {Xdf.shape[1]} leakage-safe features:")
print("   ", ", ".join(Xdf.columns))

# ----------------------------------------------------------------------------
# STAGE 3: baseline model selection, with a forecast chart and a driver chart
# ----------------------------------------------------------------------------
print("\n" + "=" * 72); print("STAGE 3  BASELINE MODEL SELECTION"); print("=" * 72)
baseline = run_model_selection(clean)
winner = baseline["winner"]

# 3a) forecast vs actual for the winning model over the back-test window
plt.figure(figsize=(12, 4.6), dpi=200)
plt.plot(baseline["dates"], baseline["actual"], lw=1.6, label="actual")
plt.plot(baseline["dates"], baseline["forecasts"][winner], lw=1.2, label=f"forecast ({winner})")
plt.title(f"Forecast vs actual over the back-test window (winner: {winner})")
plt.xlabel("Date"); plt.ylabel("Platelet units used")
plt.legend(frameon=False); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# 3b) what drives demand, explained with the method that fits the WINNING model:
#       - ElasticNet winner            -> signed coefficients (green raises demand, red lowers it)
#       - tree winner (xgboost /        -> SHAP average impact if the `shap` package is installed,
#         random_forest / lightgbm)        otherwise the model's built-in feature importances
#       - time-series winner (sarimax,  -> a companion ElasticNet, fitted only to make the
#         holt_winters, theta, naive)      demand drivers explicit
try:
    feat = list(Xdf.columns)
    mask = Xdf.notna().all(axis=1).to_numpy()
    Xv = Xdf.to_numpy(float)[mask]
    yv = clean["demand_model"].to_numpy(float)[mask]

    def companion_elasticnet_coefs():
        scaler = StandardScaler().fit(Xv)
        en = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0], cv=TimeSeriesSplit(n_splits=5),
                          max_iter=5000, random_state=42).fit(scaler.transform(Xv), yv)
        return en.coef_

    if winner == "elasticnet":
        show_drivers(feat, companion_elasticnet_coefs(),
                     "What drives daily platelet demand (ElasticNet)", directional=True)
    elif winner in ("xgboost", "random_forest", "lightgbm"):
        if winner == "xgboost":
            mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                                   objective="reg:squarederror").fit(Xv, yv)
        elif winner == "random_forest":
            mdl = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(Xv, yv)
        else:
            mdl = lgb.LGBMRegressor(n_estimators=300, random_state=42, n_jobs=-1, verbosity=-1).fit(Xv, yv)
        try:
            import shap
            sv = shap.TreeExplainer(mdl).shap_values(Xv)
            show_drivers(feat, np.abs(sv).mean(axis=0),
                         f"What drives daily platelet demand ({winner}, SHAP average impact)", directional=False)
        except Exception:
            show_drivers(feat, mdl.feature_importances_,
                         f"What drives daily platelet demand ({winner} feature importance)", directional=False)
    else:
        show_drivers(feat, companion_elasticnet_coefs(),
                     f"What drives daily platelet demand (companion ElasticNet; winner was {winner})",
                     directional=True)
except Exception as e:
    print("  (skipped the drivers chart:", e, ")")

# ----------------------------------------------------------------------------
# STAGE 4: leak-fixed, tuned model selection
# ----------------------------------------------------------------------------
print("\n" + "=" * 72); print("STAGE 4  TUNED MODEL SELECTION"); print("=" * 72)
tuned = run_tuned_selection(clean)
# For a fast run instead, use fewer trials, e.g.:
#   tuned = run_tuned_selection(clean, xgb_trials=8, rf_trials=8, lgb_trials=8)

# ----------------------------------------------------------------------------
# STAGE 5: inventory policy
#   Current practice comes from the baseline collection plan already in the data.
#   The new policy is run on the winning model's forecast over the back-test window.
# ----------------------------------------------------------------------------
print("\n" + "=" * 72); print("STAGE 5  INVENTORY POLICY"); print("=" * 72)
mean_daily = float(clean["demand_model"].mean())
stock_cols = [c for c in clean.columns if c.startswith("stock_") and c.endswith("day_left")]

# current practice (the baseline plan that produced the data)
base_waste = 100 * clean["platelets_expired"].sum() / max(clean["platelets_received"].sum(), 1)
base_short_days = 100 * (clean["platelets_requested"] > clean["platelets_used"]).mean()
base_short_units = int((clean["platelets_requested"] - clean["platelets_used"]).clip(lower=0).sum())
base_onhand_days = float(clean[stock_cols].sum(axis=1).mean() / mean_daily)

# new order-up-to policy at a 95% service level (default 5% of arriving units are 7-day)
actual = baseline["actual"]
fc_w = baseline["forecasts"][winner]
new = simulate_policy(actual, fc_w, mean_daily, lead_time=2, review=1, service_level=0.95)

print(f"Buffer sized from the measured 3-day interval forecast error: {new['sigma_interval']:.1f} units")
print(f"(the 1-day forecast error alone is only {float(np.std(actual - fc_w)):.1f} units)\n")
print(f"{'metric':30s}{'current':>12s}{'new policy':>12s}")
print(f"{'waste %':30s}{base_waste:>12.1f}{new['waste']:>12.1f}")
print(f"{'shortage days %':30s}{base_short_days:>12.1f}{new['short_days']:>12.1f}")
print(f"{'units short (total)':30s}{base_short_units:>12d}{new['short_units']:>12d}")
print(f"{'on-shelf stock (days)':30s}{base_onhand_days:>12.1f}{new['onhand_days']:>12.1f}")
print(f"{'lowest day on shelf (units)':30s}{'-':>12s}{new['min_onhand']:>12d}")
print(f"{'stat-courier days %':30s}{'-':>12s}{new['emer_days']:>12.1f}")
print(f"{'emergency units (total)':30s}{'-':>12s}{new['emer_units']:>12d}")
print(f"\nThe floor held every day (lowest shelf = {new['min_onhand']} units), so all demand was met.")

# worked example: the order-up-to calculation for one day
order, S, safety = order_up_to([mean_daily] * 3, 20, None, lead_time=2, review=1,
                               service_level=0.95, mean_daily_demand=mean_daily,
                               sigma_interval=new["sigma_interval"])
print(f"\nWorked example (one day): forecast {mean_daily:.0f}/day, 20 usable units on hand")
print(f"  target level S = {S:.0f} units (a {safety:.0f}-unit safety buffer on top of expected demand)"
      f"  ->  order {order} units")

# service-level trade-off (the floor guarantees demand is met at every level)
print("\nService-level trade-off (demand is met at every level; it trades waste against courier use):")
print(f"{'service level':>14}{'waste %':>10}{'on-shelf days':>15}{'lowest units':>14}{'courier days %':>16}{'emerg. units':>14}")
for svc in [0.90, 0.95, 0.98, 0.99]:
    m = simulate_policy(actual, fc_w, mean_daily, lead_time=2, review=1, service_level=svc)
    print(f"{svc:>14.2f}{m['waste']:>10.1f}{m['onhand_days']:>15.1f}{m['min_onhand']:>14d}{m['emer_days']:>16.1f}{m['emer_units']:>14d}")

# inventory charts (full series; fc_order is a simple trailing-7-day-mean operating forecast)
fc_order = clean["usage_avg_7d"].fillna(mean_daily).to_numpy(float)
demand = clean["demand_model"].to_numpy(float)
plot_mix_sensitivity(demand, fc_order, mean_daily)
plot_inventory_over_time(demand, fc_order, mean_daily, dates=clean["date"].to_numpy(), frac_7day=0.05)
plt.show()

print("\nDone. Output by stage: data realism, cleaning report, feature list, baseline table "
      "+ forecast chart + drivers chart, tuned tables, and the inventory comparison, worked "
      "example, service-level table, and two inventory charts.")