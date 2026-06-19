"""
Inventory policy and policy back-test for the platelet pipeline.

Run order: after the model-selection steps (forecasters.py, tuning.py). Holds the order-up-to
policy (order_up_to), its forecast-error helper (interval_error_std), and the day-by-day policy
back-test (simulate_policy), which supports a mixed 5-day / 7-day inventory. The plotting step
calls simulate_policy, so this file must run before plotting.py.

Acronyms: std = standard deviation.
"""
import numpy as np
from scipy.stats import norm


def order_up_to(demand_forecast_LR, inventory_position, sigma_1step,
                lead_time=1, review=1, service_level=0.95,
                mean_daily_demand=None, max_cover_days=5, sigma_interval=None):
    """Base-stock (order-up-to) policy for a perishable product.

    demand_forecast_LR : forecast demand for each of the next (lead_time + review) days.
    inventory_position : usable units on hand + units already collected but not yet available.
    sigma_1step        : standard deviation of the 1-day-ahead forecast error (from the back-test).
    sigma_interval     : optional. If given, the safety stock uses this measured error over the
                         whole protection interval instead of scaling the 1-day error.
    Returns (order_qty, S, safety_stock)."""
    horizon = lead_time + review
    fc = np.atleast_1d(np.asarray(demand_forecast_LR, dtype=float))
    mean_LR = float(np.sum(fc[:horizon]))                  # expected demand over the protection interval
    z = float(norm.ppf(service_level))                     # service-level multiplier
    if sigma_interval is not None:
        safety = z * sigma_interval                        # buffer from the measured interval error
    else:
        safety = z * sigma_1step * np.sqrt(horizon)        # buffer for forecast error over the interval
    S = mean_LR + safety                                   # order-up-to (target) level
    if max_cover_days is not None and mean_daily_demand:   # perishability guard (avoid guaranteed waste)
        S = min(S, max_cover_days * mean_daily_demand)
    order = max(0, int(round(S - inventory_position)))     # collect enough to reach S
    return order, S, safety


def interval_error_std(demand, fc_order, horizon):
    """Std of the model's error in forecasting total demand over the next `horizon` days."""
    errs = [demand[t+1:t+1+horizon].sum() - horizon*fc_order[t] for t in range(len(demand)-horizon)]
    return float(np.std(errs))


def simulate_policy(demand, fc_order, mean_daily, lead_time=2, review=1, service_level=0.95,
                    shelf_life=5, max_cover_days=None, safety_floor=10,
                    frac_7day=0.05, shelf_life_long=7):
    """Run the order-up-to policy forward one day at a time over a MIXED perishable inventory.

    Two pools are tracked separately, each ageing to its own expiry:
      - a SHORT-life pool (shelf_life days; the default 5-day, pathogen-reduced units), and
      - a LONG-life pool  (shelf_life_long days; the default 7-day, bacterially-tested units that
        Stanford Blood Center sometimes purchases).
    frac_7day is the share of each day's ARRIVING units that are long-life (0.0 to 1.0). Set it to
    stress-test: 0.0, 0.05 (default, the small purchased share), 0.20, and so on.

    Issuing rule: CLOSEST-TO-EXPIRY FIRST across both pools (use the most perishable unit first).
    This is the lowest-waste rule, a generalization of oldest-first issuing to two pools.

    max_cover_days is the perishability cap that stops the order-up-to target from exceeding what
    can realistically be used before it expires. It now DEFAULTS to the blended shelf life of the
    mix (shelf_life and shelf_life_long weighted by frac_7day), so the cap scales with the product
    mix instead of being fixed at 5. Pass a number to override.

    Emergency same-day courier units are treated as short-life (standard product).

    With frac_7day = 0.0 every long-life quantity is zero, so this collapses to a single 5-day pool
    and the single-pool metrics are recovered. Returns those metrics PLUS a per-pool waste
    breakdown (expired_5day/7day, received_5day/7day, waste_5day/7day, frac_7day)."""
    demand = np.asarray(demand, dtype=float)
    fc_order = np.asarray(fc_order, dtype=float)
    n = len(demand)
    S_SHORT = int(shelf_life); S_LONG = int(shelf_life_long)
    horizon = lead_time + review
    sigma_interval = interval_error_std(demand, fc_order, horizon)

    # Perishability cap scales with the mix. At frac_7day = 0 this equals shelf_life exactly,
    # so the pure 5-day behaviour is preserved.
    blended_shelf = shelf_life * (1.0 - frac_7day) + shelf_life_long * frac_7day
    cover_cap = blended_shelf if max_cover_days is None else max_cover_days

    # stock_s[k] = short-life units with (k+1) days left; stock_l[k] = long-life units, same idea.
    entry_s = max(S_SHORT - lead_time - 1, 0)        # arrives with (shelf_life - lead_time) days left
    entry_l = max(S_LONG - lead_time - 1, 0)         # arrives with (shelf_life_long - lead_time) days left
    stock_s = [0]*S_SHORT
    stock_l = [0]*S_LONG

    # Seed steady operation, split by the mix (a single 5-day seed when frac_7day=0).
    seed = int(round(mean_daily))
    seed_l = int(round(seed * frac_7day)); seed_s = seed - seed_l
    stock_s[entry_s] += seed_s
    stock_l[entry_l] += seed_l
    pipeline = [int(round(mean_daily))]*lead_time    # orders already collected, in testing (totals)

    rec_s = rec_l = exp_s = exp_l = emer_units = emer_days = 0
    short = []; onhand = []; emer_flag = []
    onhand_s = []; onhand_l = []                          # per-pool end-of-day stock, for the inventory chart

    for t in range(n):
        # 1) units that reached expiry are discarded (per pool), then everything ages one day
        exp_s += stock_s[0]; exp_l += stock_l[0]
        for k in range(S_SHORT-1): stock_s[k] = stock_s[k+1]
        stock_s[S_SHORT-1] = 0
        for k in range(S_LONG-1): stock_l[k] = stock_l[k+1]
        stock_l[S_LONG-1] = 0

        # 2) the order placed `lead_time` days ago arrives and is split into the two pools
        arrive = pipeline.pop(0)
        arrive_l = int(round(arrive * frac_7day)); arrive_s = arrive - arrive_l
        stock_s[entry_s] += arrive_s; stock_l[entry_l] += arrive_l
        rec_s += arrive_s; rec_l += arrive_l

        # 3) routine collection from the order-up-to target (one shared order across both pools)
        ip = sum(stock_s) + sum(stock_l) + sum(pipeline)     # inventory position = shelf + testing pipeline
        order, S, _ = order_up_to([fc_order[t]]*horizon, ip, None, lead_time, review, service_level,
                                  mean_daily, cover_cap, sigma_interval=sigma_interval)
        pipeline.append(order)                               # usable in `lead_time` days

        # 4) emergency same-day courier if the safety floor would break (standard short-life product)
        d = int(round(demand[t])); on_shelf = sum(stock_s) + sum(stock_l)
        emerg = max(0, safety_floor + d - on_shelf)
        if emerg > 0:
            stock_s[entry_s] += emerg; emer_units += emerg; emer_days += 1
        emer_flag.append(emerg > 0)

        # 5) serve demand CLOSEST-TO-EXPIRY FIRST across both pools (lowest days-left first)
        need = d; served = 0
        for dleft in range(1, max(S_SHORT, S_LONG) + 1):
            if need == 0: break
            if dleft <= S_SHORT:
                take = min(stock_s[dleft-1], need); stock_s[dleft-1] -= take; served += take; need -= take
            if need == 0: break
            if dleft <= S_LONG:
                take = min(stock_l[dleft-1], need); stock_l[dleft-1] -= take; served += take; need -= take
        short.append(d - served)
        s5_now = sum(stock_s); s7_now = sum(stock_l)
        onhand.append(s5_now + s7_now); onhand_s.append(s5_now); onhand_l.append(s7_now)

    short = np.array(short); onhand = np.array(onhand)
    received = rec_s + rec_l; expired = exp_s + exp_l
    return dict(waste=expired/max(received + emer_units, 1)*100, expired=int(expired),
                short_days=float(np.mean(short>0)*100), short_units=int(short.sum()),
                onhand_days=float(onhand.mean()/mean_daily), onhand_series=onhand, min_onhand=int(onhand.min()),
                emer_days=float(emer_days/n*100), emer_count=int(emer_days), emer_units=int(emer_units),
                emer_flag=np.array(emer_flag), sigma_interval=sigma_interval,
                # ---- per-pool breakdown (new for the 5-day / 7-day mix) ----
                frac_7day=float(frac_7day),
                expired_5day=int(exp_s), expired_7day=int(exp_l),
                received_5day=int(rec_s), received_7day=int(rec_l),
                waste_5day=exp_s/max(rec_s + emer_units, 1)*100,
                waste_7day=exp_l/max(rec_l, 1)*100,
                onhand_5day_series=np.array(onhand_s), onhand_7day_series=np.array(onhand_l))
