"""
Charts for the platelet pipeline.

Run order: last. Holds the driver chart (show_drivers) and the two policy charts
(plot_mix_sensitivity, plot_inventory_over_time). The two policy charts call simulate_policy from
inventory_policy.py, so run that file first. Every chart takes a dpi argument and an optional
savepath for high-resolution output.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = {
    "lag_1": "Yesterday's use", "lag_7": "Same weekday last week", "lag_14": "Use two weeks ago",
    "roll7_mean": "Avg use, last 7 days", "roll7_std": "Last week's ups and downs",
    "dow_1": "Tuesday", "dow_2": "Wednesday", "dow_3": "Thursday",
    "dow_4": "Friday", "dow_5": "Saturday", "dow_6": "Sunday",
    "is_holiday": "Holiday",
    "sin_year": "Season: spring vs autumn", "cos_year": "Season: winter vs summer",
    "scheduled_surgeries": "Scheduled surgeries",
    "census_heme_onc_lag1": "Cancer-ward patients (yesterday)",
    "census_icu_lag1": "ICU patients (yesterday)",
    "census_med_surg_lag1": "Med-surg patients (yesterday)",
    "cbc_low_platelet_lag1": "Low-platelet labs (yesterday)",
    "cbc_abnormal_mcv_lag1": "Abnormal MCV labs (yesterday)",
    "cbc_high_rdw_lag1": "High RDW labs (yesterday)",
}


def pretty(name):
    """Return a human-readable label for a feature name, using the LABELS map with a safe fallback for any name not listed."""
    if name in LABELS:
        return LABELS[name]
    s = name.replace("_lag1", " (yesterday)").replace("_", " ")   # safe fallback for any new feature
    return s[:1].upper() + s[1:]


def show_drivers(names, values, title, directional=True, top=12, savepath=None, dpi=200):
    """Horizontal bar chart of the model's drivers. `values` are per-feature driver scores supplied
    by the caller: signed model coefficients or signed SHAP values (use directional=True), or
    unsigned feature importances or mean absolute SHAP impacts (use directional=False). With
    directional=True the bars are signed effects on demand (green positive, red negative); otherwise
    they are unsigned importances. The two seasonal Fourier terms are merged into one season bar
    whose peak season is read from the fitted coefficients. Pass savepath to save a high-resolution
    image, otherwise the chart is shown."""
    names = list(names); values = list(np.asarray(values, dtype=float))
    # Season is two math terms (sin_year, cos_year). Merge them into ONE plain bar,
    # and read the peak season straight from the fitted model so the label is always correct.
    if "sin_year" in names and "cos_year" in names:
        i_s = names.index("sin_year"); i_c = names.index("cos_year")
        b_s = values[i_s]; b_c = values[i_c]
        if directional:
            R = float(np.hypot(b_s, b_c))                  # combined strength (same scale as the other bars)
            # Only name a season if the seasonal signal is real. If it is negligible next to the other
            # drivers, the peak angle is just noise, so label it plainly instead of naming a false season.
            other = [abs(v) for j, v in enumerate(values) if j not in (i_s, i_c)]
            ref = max(other) if other else 0.0
            if R <= 1e-9 or (ref > 0 and R < 0.05 * ref):
                season_label = "Season (negligible)"
            else:
                theta = float(np.arctan2(b_s, b_c))        # angle of the yearly peak
                if theta < 0: theta += 2*np.pi
                doy = theta/(2*np.pi)*365.0                 # day of year where demand peaks
                season = ("winter" if (doy >= 335 or doy < 60) else
                          "spring" if doy < 152 else
                          "summer" if doy < 244 else "autumn")
                season_label = "Higher demand in " + season
            season_val = R
        else:
            season_label = "Season (time of year)"; season_val = float(b_s + b_c)
        for i in sorted([i_s, i_c], reverse=True):
            del names[i]; del values[i]
        names.append(season_label); values.append(season_val)
    values = np.asarray(values, dtype=float)
    idx = np.argsort(np.abs(values))[::-1][:top][::-1]      # strongest at the top of the chart
    names = [pretty(names[i]) for i in idx]; vals = [float(values[i]) for i in idx]
    cols = (["seagreen" if v >= 0 else "indianred" for v in vals]) if directional else (["steelblue"]*len(vals))
    fig, ax = plt.subplots(figsize=(8.5, 4.3), dpi=dpi)
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    # Label the axis correctly for each mode: a directional chart shows the signed effect on demand,
    # an importance chart shows relative importance (no direction).
    ax.set_xlabel("Effect on daily platelet demand (units)" if directional else "Relative importance")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight"); plt.close(fig); return savepath
    plt.show()


def plot_mix_sensitivity(demand, fc_order, mean_daily, fracs=None, savepath=None, dpi=200, **policy_kwargs):
    """Sweep the 7-day share and show how waste and shortages respond, on the SAME demand+forecast.

    Runs simulate_policy once per value in `fracs` (default 0% to 30% in 5-point steps) and draws
    two panels:
      left  - waste, as a percent of units collected: overall, plus the 5-day and 7-day pools;
      right - service risk: percent of days with a shortage, and percent of days needing an
              emergency courier.
    A dotted line marks the default 5% share. Pass savepath to save a PNG, otherwise the chart is
    shown. Extra keywords (lead_time, service_level, safety_floor, ...) pass through to simulate_policy.
    Returns the swept values as a dict so the numbers can be reused."""
    import matplotlib.ticker as mticker
    if fracs is None:
        fracs = [i/100 for i in range(0, 31, 5)]            # 0%, 5%, ... 30%
    fracs = list(fracs)
    waste = []; waste5 = []; waste7 = []; short = []; emer = []
    for f in fracs:
        r = simulate_policy(demand, fc_order, mean_daily, frac_7day=f, **policy_kwargs)
        waste.append(r["waste"]); waste5.append(r["waste_5day"]); waste7.append(r["waste_7day"])
        short.append(r["short_days"]); emer.append(r["emer_days"])
    x = np.asarray(fracs) * 100.0                            # plot the share as a percent

    fig, (axw, axs) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=dpi)

    # Left panel: waste, overall and per pool.
    axw.plot(x, waste,  "-o", color="#1f77b4", lw=2.0, label="All units")
    axw.plot(x, waste5, "--s", color="#2ca02c", lw=1.6, label="5-day pool")
    axw.plot(x, waste7, "--^", color="#d62728", lw=1.6, label="7-day pool")
    axw.set_title("Waste by 7-day share")
    axw.set_xlabel("Share of arriving units that are 7-day (%)")
    axw.set_ylabel("Units expired (% of units collected)")
    axw.legend(frameon=False, fontsize=9)
    axw.grid(alpha=0.3)

    # Right panel: service risk.
    axs.plot(x, short, "-o", color="#9467bd", lw=2.0, label="Days with a shortage")
    axs.plot(x, emer,  "--d", color="#ff7f0e", lw=1.6, label="Days needing emergency courier")
    axs.set_title("Shortage risk by 7-day share")
    axs.set_xlabel("Share of arriving units that are 7-day (%)")
    axs.set_ylabel("Days (% of all days)")
    axs.legend(frameon=False, fontsize=9)
    axs.grid(alpha=0.3)
    # Make the zero shortage line self-explanatory: the courier is the backstop that prevents it.
    axs.text(0.04, 0.5,
             "Shortages stay near zero:\nthe safety floor fires an\nemergency courier instead.",
             transform=axs.transAxes, fontsize=8, color="dimgray", va="center",
             bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    for ax in (axw, axs):
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.axvline(5, color="gray", ls=":", lw=1.2)         # the default 5% share
        ax.text(5, 0.98, " default 5%", transform=ax.get_xaxis_transform(),
                fontsize=8, color="gray", ha="left", va="top")

    fig.suptitle("Platelet inventory: sensitivity to the 7-day share", fontsize=13, y=1.02)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight"); plt.close(fig); return savepath
    plt.show()
    return dict(frac_7day=fracs, waste=waste, waste_5day=waste5, waste_7day=waste7,
                short_days=short, emer_days=emer)


def plot_inventory_over_time(demand, fc_order, mean_daily, days=365, start=0, dates=None,
                             savepath=None, frac_7day=0.05, dpi=200, **policy_kwargs):
    """Run the policy and chart END-OF-DAY on-hand stock, stacked into the 5-day and 7-day pools.

    Shows a `days`-long window (default one year) starting at row `start`. If `dates` is given
    (a date-like array aligned to demand), the x-axis uses real dates; otherwise it uses day number.
    frac_7day sets the 7-day share (default 0.05). The top of the stack is the total on hand.
    Pass savepath to save a PNG, otherwise the chart is shown. Extra keywords pass through to
    simulate_policy. Returns the two plotted series so the numbers can be reused."""
    r = simulate_policy(demand, fc_order, mean_daily, frac_7day=frac_7day, **policy_kwargs)
    s5 = np.asarray(r["onhand_5day_series"], dtype=float)
    s7 = np.asarray(r["onhand_7day_series"], dtype=float)
    n = len(s5)
    a = max(0, int(start)); b = min(n, a + int(days))
    s5 = s5[a:b]; s7 = s7[a:b]
    x = pd.to_datetime(np.asarray(dates)[a:b]) if dates is not None else np.arange(a, b)

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=dpi)
    ax.stackplot(x, s5, s7, labels=["5-day pool", "7-day pool"],
                 colors=["#2ca02c", "#d62728"], alpha=0.85)
    ax.plot(x, s5 + s7, color="#1f77b4", lw=1.4, label="Total on hand")
    ax.set_title(f"Daily platelet stock by pool ({frac_7day*100:.0f}% of arriving units are 7-day)")
    ax.set_xlabel("Date" if dates is not None else "Day")
    ax.set_ylabel("Units on hand (end of day)")
    # Legend outside, to the right, so it never overlaps the tall stock peaks.
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=9)
    ax.grid(alpha=0.3); ax.margins(x=0)
    if dates is not None:
        fig.autofmt_xdate()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight"); plt.close(fig); return savepath
    plt.show()
    return dict(onhand_5day=s5, onhand_7day=s7)