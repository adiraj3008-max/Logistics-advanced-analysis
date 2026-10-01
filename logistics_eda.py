#!/usr/bin/env python3
"""
Advanced Data Analysis and Visualization in Logistics
=====================================================
Yuva Intern - Logistics Data Analyst Intern - Week 3 Task

Pipeline
    1. Simulate a hypothetical logistics dataset (5,000 shipments, seed = 42)
    2. Inject realistic data-quality problems, then clean the data
    3. Exploratory Data Analysis (central tendency, distributions, correlations)
    4. Nine visualizations saved to ./figures
    5. Statistical tests, driver (regression) analysis and what-if scenarios
    6. Save cleaned data to ./data and all numbers to results.json

Run:  python logistics_eda.py
Requires: numpy, pandas, matplotlib, seaborn, scipy
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42
N_SHIPMENTS = 5000

NAVY, ORANGE, TEAL, RED, GREY = "#1F3A5F", "#F28C28", "#2A9D8F", "#D1495B", "#6B7280"
MODE_COLORS = {"Road": NAVY, "Rail": TEAL, "Air": ORANGE}
MODES = ["Road", "Rail", "Air"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 12.5,
    "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
})

# ----------------------------------------------------------------------------
# 1. DATA SIMULATION
# ----------------------------------------------------------------------------
HUBS = ["Delhi", "Mumbai", "Bengaluru", "Kolkata", "Nagpur"]
HUB_REGION = {"Delhi": "North", "Mumbai": "West", "Bengaluru": "South",
              "Kolkata": "East", "Nagpur": "Central"}
REGION_HUB = {v: k for k, v in HUB_REGION.items()}
ROAD_KM = {("Delhi", "Mumbai"): 1400, ("Delhi", "Bengaluru"): 2150,
           ("Delhi", "Kolkata"): 1500, ("Delhi", "Nagpur"): 1050,
           ("Mumbai", "Bengaluru"): 980, ("Mumbai", "Kolkata"): 2000,
           ("Mumbai", "Nagpur"): 830, ("Bengaluru", "Kolkata"): 1870,
           ("Bengaluru", "Nagpur"): 1000, ("Kolkata", "Nagpur"): 1100}


def hub_distance(a, b):
    return ROAD_KM.get((a, b), ROAD_KM.get((b, a)))


def simulate_dataset(n=N_SHIPMENTS, seed=SEED):
    """Generate a hypothetical shipment-level logistics dataset."""
    rng = np.random.default_rng(seed)
    # [[SIM]]
    # --- when, where and what is shipped ---------------------------------
    month_w = np.array([.075, .07, .08, .075, .075, .07, .07, .08, .085, .10, .11, .11])
    month = rng.choice(np.arange(1, 13), size=n, p=month_w / month_w.sum())
    order_date = pd.to_datetime(dict(year=2025, month=month, day=rng.integers(1, 29, size=n)))
    peak = np.isin(month, [10, 11, 12])                      # Oct-Dec festive peak

    hub = rng.choice(HUBS, size=n, p=[.24, .28, .22, .14, .12])
    dest = rng.choice(["North", "West", "South", "East", "Central"], size=n,
                      p=[.22, .25, .24, .17, .12])
    dest_hub = np.array([REGION_HUB[r] for r in dest])
    same = hub == dest_hub
    inter_km = np.array([0 if s else hub_distance(h, d) for h, d, s in zip(hub, dest_hub, same)], float)
    distance = np.where(same, rng.uniform(40, 350, n), inter_km * rng.uniform(0.92, 1.15, n))

    weight = np.clip(rng.lognormal(np.log(220), 0.95, n), 5, 6000)         # right-skewed
    priority = rng.choice(["Standard", "Express"], size=n, p=[.75, .25])
    carrier = rng.choice(["SwiftMove", "BlueLine", "TransGo", "RapidFreight"], size=n,
                         p=[.32, .28, .22, .18])

    u = rng.random(n)                                                       # choose transport mode
    mode = np.where((priority == "Express") & (distance > 700), np.where(u < .40, "Air", "Road"),
                    np.where((priority == "Standard") & (distance > 500),
                             np.where(u < .32, "Rail", "Road"), "Road"))

    weather = np.empty(n, dtype=object)                                     # seasonal weather
    for m in range(1, 13):
        idx = month == m
        p = [.50, .38, .10, .02] if m in (6, 7, 8, 9) else \
            [.72, .05, .01, .22] if m in (12, 1) else [.82, .12, .03, .03]
        weather[idx] = rng.choice(["Clear", "Rain", "Storm", "Fog"], size=idx.sum(), p=p)

    traffic = np.clip(rng.normal(5, 1.4, n) + 1.0 * peak + 1.2 * (hub == "Mumbai")
                      + 0.6 * (hub == "Delhi"), 1, 10)                      # congestion index 1-10

    # --- delivery time (hours) = hub handling + transit + disruptions ----
    speed = pd.Series(mode).map({"Road": 48, "Rail": 36, "Air": 650}).to_numpy()
    terminal = pd.Series(mode).map({"Road": 0, "Rail": 14, "Air": 12}).to_numpy()
    express_road = np.where((priority == "Express") & (mode == "Road"), 0.82, 1.0)
    transit = (distance / speed) * express_road + terminal
    hub_base = pd.Series(hub).map({"Delhi": 7, "Mumbai": 9, "Bengaluru": 6.5,
                                   "Kolkata": 8, "Nagpur": 5.5}).to_numpy()
    handling = (hub_base + weight / 600 + rng.exponential(1.5, n)
                + 6 * ((hub == "Mumbai") & peak) + 2 * ((hub == "Delhi") & peak))
    carrier_delay = pd.Series(carrier).map({"SwiftMove": 0.0, "BlueLine": 1.5,
                                            "TransGo": 6.0, "RapidFreight": -1.5}).to_numpy()
    weather_delay = np.select(
        [weather == "Rain", weather == "Storm", weather == "Fog"],
        [np.clip(rng.normal(5, 2, n), 0, None), np.clip(rng.normal(16, 5, n), 0, None),
         np.clip(rng.normal(7, 3, n), 0, None)], default=0.0)
    traffic_delay = np.where(mode == "Road", 0.9 * (traffic - 5), 0.0)
    delivery_time = (handling + transit * rng.lognormal(0, 0.10, n) + weather_delay
                     + carrier_delay * rng.uniform(0.6, 1.4, n) + traffic_delay)
    delivery_time = np.clip(delivery_time, 3, None)
    promised = (hub_base + transit) * 1.15 + 6                              # SLA promise
    delay_hours = np.clip(delivery_time - promised, 0, None)

    # --- transport cost (USD) ---------------------------------------------
    fuel_by_month = np.array([98, 99, 101, 103, 104, 102, 101, 103, 105, 108, 110, 111]) / 100
    fuel_index = fuel_by_month[month - 1] + rng.normal(0, 0.01, n)
    fixed = pd.Series(mode).map({"Road": 12, "Rail": 25, "Air": 40}).to_numpy()
    per_km = pd.Series(mode).map({"Road": 0.12, "Rail": 0.07, "Air": 0.10}).to_numpy()
    per_kg = pd.Series(mode).map({"Road": 0.06, "Rail": 0.04, "Air": 0.85}).to_numpy()
    carrier_mult = pd.Series(carrier).map({"SwiftMove": 1.00, "BlueLine": 0.94,
                                           "TransGo": 0.88, "RapidFreight": 1.18}).to_numpy()
    fuel_sens = np.where(mode == "Rail", 0.2, 0.6)
    cost = (fixed + per_km * distance + per_kg * weight)
    cost = (cost * np.where(priority == "Express", 1.20, 1.0) * carrier_mult
            * (1 + fuel_sens * (fuel_index - 1)) * (1 + 0.08 * (weather == "Storm"))
            * rng.lognormal(0, 0.08, n))

    rating = np.clip(np.round(4.7 - 0.06 * delay_hours + rng.normal(0, 0.55, n)), 1, 5)
    # [[/SIM]]
    df = pd.DataFrame({
        "shipment_id": [f"SH{i:05d}" for i in range(1, n + 1)],
        "order_date": order_date, "origin_hub": hub, "destination_region": dest,
        "lane": [f"{h} \u2192 {d}" for h, d in zip(hub, dest)],
        "carrier": carrier, "transport_mode": mode, "priority": priority, "weather": weather,
        "weight_kg": weight.round(1), "distance_km": distance.round(1),
        "traffic_index": traffic.round(2), "fuel_index": fuel_index.round(3),
        "delivery_time_hrs": delivery_time.round(2), "promised_time_hrs": promised.round(2),
        "delay_hours": delay_hours.round(2), "delayed": (delay_hours > 0).astype(int),
        "transport_cost_usd": cost.round(2), "customer_rating": rating,
    })
    return df.sort_values("order_date").reset_index(drop=True).assign(
        shipment_id=lambda d: [f"SH{i:05d}" for i in range(1, len(d) + 1)])


def inject_data_issues(df, seed=SEED):
    """Add realistic quality problems (missing values, duplicates) to practise cleaning."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    df.loc[rng.choice(df.index, 60, replace=False), "weight_kg"] = np.nan        # missing weights
    df.loc[rng.random(len(df)) < 0.05, "customer_rating"] = np.nan               # unrated deliveries
    dup = df.sample(30, random_state=seed)                                       # double-posted rows
    return pd.concat([df, dup], ignore_index=True)


# ----------------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------------
def clean_dataset(raw):
    # [[CLEAN]]
    report = {"rows_raw": int(len(raw)), "columns": int(raw.shape[1])}
    df = raw.drop_duplicates(subset="shipment_id").copy()                  # 1) duplicates
    report["duplicates_removed"] = int(len(raw) - len(df))
    report["missing_weight"] = int(df["weight_kg"].isna().sum())           # 2) missing values
    report["missing_rating"] = int(df["customer_rating"].isna().sum())
    df["weight_kg"] = df["weight_kg"].fillna(
        df.groupby("transport_mode")["weight_kg"].transform("median"))     #    median impute
    df["order_date"] = pd.to_datetime(df["order_date"])                    # 3) types + features
    df["month"] = df["order_date"].dt.month
    df["is_peak"] = df["month"].isin([10, 11, 12])
    df["cost_per_kg"] = df["transport_cost_usd"] / df["weight_kg"]
    df["cost_per_kg_km"] = df["transport_cost_usd"] / (df["weight_kg"] * df["distance_km"])
    for col in ["weight_kg", "transport_cost_usd", "delivery_time_hrs"]:  # 4) IQR outlier audit
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        report[f"outliers_{col}"] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    report["rows_clean"] = int(len(df))
    # [[/CLEAN]]
    return df.reset_index(drop=True), report


# ----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
NUM_LABELS = {
    "weight_kg": "Shipment weight (kg)", "distance_km": "Distance (km)",
    "delivery_time_hrs": "Delivery time (hrs)", "transport_cost_usd": "Transport cost (USD)",
    "cost_per_kg": "Cost per kg (USD)", "traffic_index": "Traffic index (1-10)",
    "customer_rating": "Customer rating (1-5)",
}


def summary_statistics(df):
    # [[EDA]]
    rows = []
    for col, label in NUM_LABELS.items():
        x = df[col].dropna()
        counts, edges = np.histogram(x, bins=30)
        k = counts.argmax()
        rows.append({
            "variable": label, "mean": x.mean(), "median": x.median(),
            "mode": (edges[k] + edges[k + 1]) / 2, "std": x.std(),
            "min": x.min(), "max": x.max(), "cv_pct": x.std() / x.mean() * 100,
            "skew": stats.skew(x),
        })
    return pd.DataFrame(rows)
    # [[/EDA]]


# ----------------------------------------------------------------------------
# 4. VISUALIZATIONS
# ----------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig1_distributions(df):
    # [[FIG1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    t = df["delivery_time_hrs"]
    sns.histplot(t, bins=40, kde=True, color=NAVY, ax=axes[0])
    axes[0].axvline(t.mean(), color=RED, ls="--", lw=1.8, label=f"Mean = {t.mean():.1f} h")
    axes[0].axvline(t.median(), color=ORANGE, ls="-.", lw=1.8, label=f"Median = {t.median():.1f} h")
    axes[0].set(title="(a) Delivery time is right-skewed", xlabel="Delivery time (hours)", ylabel="Shipments")
    axes[0].legend(frameon=False)
    w = df["weight_kg"]
    sns.histplot(w, bins=45, log_scale=True, kde=True, color=TEAL, ax=axes[1])
    axes[1].axvline(w.median(), color=ORANGE, ls="-.", lw=1.8, label=f"Median = {w.median():.0f} kg")
    axes[1].set(title="(b) Shipment weight (log scale)", xlabel="Weight (kg, log scale)", ylabel="Shipments")
    axes[1].legend(frameon=False)
    save(fig, "fig1_distributions.png")
    # [[/FIG1]]


def fig2_boxplots(df):
    # [[FIG2]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.boxplot(data=df, x="transport_mode", y="delivery_time_hrs", order=MODES, hue="transport_mode",
                palette=MODE_COLORS, legend=False, fliersize=2.5, linewidth=1.1, ax=axes[0])
    axes[0].set(title="(a) Delivery time by transport mode", xlabel="", ylabel="Delivery time (hours)")
    sns.boxplot(data=df, x="transport_mode", y="cost_per_kg", order=MODES, hue="transport_mode",
                palette=MODE_COLORS, legend=False, fliersize=2.5, linewidth=1.1, ax=axes[1])
    axes[1].set(title="(b) Cost per kg by transport mode (log scale)", xlabel="",
                ylabel="Cost per kg (USD, log scale)", yscale="log")
    save(fig, "fig2_boxplots_by_mode.png")
    # [[/FIG2]]


def fig3_correlation(df):
    # [[FIG3]]
    cols = {"distance_km": "Distance", "weight_kg": "Weight", "traffic_index": "Traffic index",
            "fuel_index": "Fuel index", "delivery_time_hrs": "Delivery time",
            "delay_hours": "Delay hours", "transport_cost_usd": "Transport cost",
            "customer_rating": "Customer rating"}
    corr = df[list(cols)].corr().rename(index=cols, columns=cols)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=.8, linecolor="white", cbar_kws={"label": "Pearson r", "shrink": .8}, ax=ax)
    ax.set_title("Correlation matrix of key logistics variables")
    save(fig, "fig3_correlation_heatmap.png")
    # [[/FIG3]]
    return corr


def fig4_monthly_trend(df):
    # [[FIG4]]
    m = df.groupby("month").agg(shipments=("shipment_id", "count"), avg_cost=("transport_cost_usd", "mean"),
                                delay_rate=("delayed", "mean"))
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    a1.bar(m.index, m["shipments"], color="#9DB4D0", label="Shipments")
    a1.set_ylabel("Shipments per month")
    b1 = a1.twinx()
    b1.plot(m.index, m["avg_cost"], color=ORANGE, marker="o", lw=2.2, label="Average cost (USD)")
    b1.set_ylabel("Average cost per shipment (USD)")
    b1.grid(False)
    b1.spines["right"].set_visible(True)
    a1.set_title("(a) Monthly shipment volume and average transport cost")
    a2.plot(m.index, m["delay_rate"] * 100, color=RED, marker="o", lw=2.2)
    a2.axvspan(9.5, 12.5, color=ORANGE, alpha=.13, label="Peak season (Oct-Dec)")
    a2.set(title="(b) Share of shipments delivered late", ylabel="Delayed shipments (%)")
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(names)
    a2.legend(frameon=False, loc="upper left")
    save(fig, "fig4_monthly_trends.png")
    # [[/FIG4]]
    return m


def fig5_distance_vs_time(df):
    # [[FIG5]]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for mode in MODES:
        s = df[df["transport_mode"] == mode]
        ax.scatter(s["distance_km"], s["delivery_time_hrs"], s=12, alpha=.35, color=MODE_COLORS[mode],
                   label=f"{mode} (n={len(s):,})")
        slope, icpt = np.polyfit(s["distance_km"], s["delivery_time_hrs"], 1)
        xs = np.linspace(s["distance_km"].min(), s["distance_km"].max(), 50)
        ax.plot(xs, slope * xs + icpt, color=MODE_COLORS[mode], lw=2.4)
    ax.set(title="Delivery time versus distance, by transport mode",
           xlabel="Distance (km)", ylabel="Delivery time (hours)")
    ax.legend(frameon=False, markerscale=2)
    save(fig, "fig5_distance_vs_time.png")
    # [[/FIG5]]


def fig6_hub_carrier_heatmap(df):
    # [[FIG6]]
    pv = df.pivot_table(index="origin_hub", columns="carrier", values="delayed", aggfunc="mean") * 100
    pv = pv.loc[HUBS, ["RapidFreight", "SwiftMove", "BlueLine", "TransGo"]]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=.8, linecolor="white",
                cbar_kws={"label": "Delayed shipments (%)"}, ax=ax)
    ax.set(title="Delay rate (%) by origin hub and carrier", xlabel="Carrier", ylabel="Origin hub")
    save(fig, "fig6_hub_carrier_heatmap.png")
    # [[/FIG6]]
    return pv


def fig7_weather_priority(df):
    # [[FIG7]]
    order = ["Clear", "Rain", "Fog", "Storm"]
    d = df.groupby(["weather", "priority"])["delayed"].mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=d, x="weather", y="delayed", hue="priority", order=order,
                palette={"Standard": NAVY, "Express": ORANGE}, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set(title="Delay rate by weather condition and service priority",
           xlabel="Weather on the day of dispatch", ylabel="Delayed shipments (%)")
    ax.legend(title="Priority", frameon=False)
    save(fig, "fig7_weather_priority.png")
    # [[/FIG7]]
    return d


def standardized_ols(X, y):
    """OLS on z-scored predictors so coefficients are comparable (effect of +1 SD, in SD of y)."""
    Xz = (X - X.mean()) / X.std()
    yz = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(Xz)), Xz.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(A, yz.to_numpy(float), rcond=None)
    resid = yz.to_numpy(float) - A @ beta
    r2 = 1 - (resid ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return pd.Series(beta[1:], index=X.columns), float(r2)


def driver_models(df):
    # [[REG]]
    cost_X = pd.DataFrame({
        "Distance (km)": df["distance_km"], "Weight (kg)": df["weight_kg"],
        "Mode: Air": (df["transport_mode"] == "Air").astype(float),
        "Mode: Rail": (df["transport_mode"] == "Rail").astype(float),
        "Express priority": (df["priority"] == "Express").astype(float),
        "Fuel index": df["fuel_index"], "Storm day": (df["weather"] == "Storm").astype(float),
        "Carrier: RapidFreight": (df["carrier"] == "RapidFreight").astype(float),
    })
    time_X = pd.DataFrame({
        "Distance (km)": df["distance_km"], "Weight (kg)": df["weight_kg"],
        "Mode: Air": (df["transport_mode"] == "Air").astype(float),
        "Mode: Rail": (df["transport_mode"] == "Rail").astype(float),
        "Traffic index": df["traffic_index"],
        "Weather: Storm": (df["weather"] == "Storm").astype(float),
        "Weather: Rain": (df["weather"] == "Rain").astype(float),
        "Weather: Fog": (df["weather"] == "Fog").astype(float),
        "Carrier: TransGo": (df["carrier"] == "TransGo").astype(float),
        "Carrier: RapidFreight": (df["carrier"] == "RapidFreight").astype(float),
        "Hub: Mumbai": (df["origin_hub"] == "Mumbai").astype(float),
        "Peak season": df["is_peak"].astype(float),
    })
    cost_beta, cost_r2 = standardized_ols(cost_X, df["transport_cost_usd"])
    time_beta, time_r2 = standardized_ols(time_X, df["delivery_time_hrs"])
    # [[/REG]]
    return cost_beta, cost_r2, time_beta, time_r2


def fig8_drivers(cost_beta, cost_r2, time_beta, time_r2):
    # [[FIG8]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, beta, r2, title in [(axes[0], cost_beta, cost_r2, "(a) Drivers of transport cost"),
                                (axes[1], time_beta, time_r2, "(b) Drivers of delivery time")]:
        top = beta.reindex(beta.abs().sort_values(ascending=False).index).head(8)[::-1]
        ax.barh(top.index, top.values, color=[TEAL if v > 0 else RED for v in top.values])
        ax.axvline(0, color=GREY, lw=1)
        ax.set(title=f"{title}  (R\u00b2 = {r2:.2f})", xlabel="Standardised coefficient")
    save(fig, "fig8_driver_analysis.png")
    # [[/FIG8]]


def fig9_pareto(df):
    # [[FIG9]]
    lane = df.groupby("lane")["delay_hours"].sum().sort_values(ascending=False)
    cum = lane.cumsum() / lane.sum() * 100
    top = lane.head(10)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(range(len(top)), top.values, color=NAVY)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=35, ha="right")
    ax.set(title="Pareto chart: lanes ranked by total delay hours", ylabel="Total delay hours")
    ax2 = ax.twinx()
    ax2.plot(range(len(top)), cum.head(10).values, color=ORANGE, marker="o", lw=2.2)
    ax2.axhline(80, color=GREY, ls=":", lw=1.2)
    ax2.set(ylabel="Cumulative share of delay hours (%)", ylim=(0, 105))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    save(fig, "fig9_pareto_delay_lanes.png")
    # [[/FIG9]]
    return lane, cum


# ----------------------------------------------------------------------------
# 5. STATISTICAL TESTS AND WHAT-IF SCENARIOS
# ----------------------------------------------------------------------------
def run_tests(df):
    # [[TESTS]]
    out = {}
    out["chi_carrier"] = stats.chi2_contingency(pd.crosstab(df["carrier"], df["delayed"]))[:3]
    out["chi_weather"] = stats.chi2_contingency(pd.crosstab(df["weather"], df["delayed"]))[:3]
    out["chi_peak"] = stats.chi2_contingency(pd.crosstab(df["is_peak"], df["delayed"]))[:3]
    out["kruskal_cpk"] = stats.kruskal(*[g["cost_per_kg"] for _, g in df.groupby("transport_mode")])
    out["spearman_dist_time"] = stats.spearmanr(df["distance_km"], df["delivery_time_hrs"])
    out["pearson_dist_cost"] = stats.pearsonr(df["distance_km"], df["transport_cost_usd"])
    rated = df.dropna(subset=["customer_rating"])
    out["mwu_rating"] = stats.mannwhitneyu(rated.loc[rated["delayed"] == 1, "customer_rating"],
                                           rated.loc[rated["delayed"] == 0, "customer_rating"])
    out["normal_time"] = stats.normaltest(df["delivery_time_hrs"])
    out["normal_logw"] = stats.normaltest(np.log(df["weight_kg"]))
    # [[/TESTS]]
    return out


def scenarios(df):
    """Quantify four what-if improvement levers on the simulated data."""
    # [[SCEN]]
    res = {}
    total_delays = int(df["delayed"].sum())
    # 1) carrier: bring the weakest carrier to the average rate of the other carriers
    rate = df.groupby("carrier")["delayed"].mean()
    worst = rate.idxmax()
    others = df.loc[df["carrier"] != worst, "delayed"].mean()
    n_worst = int((df["carrier"] == worst).sum())
    avoided = n_worst * (rate[worst] - others)
    res["carrier"] = {"worst": worst, "rate_worst": rate[worst], "rate_others": others,
                      "shipments": n_worst, "avoided": avoided, "share_of_all": avoided / total_delays}
    # 2) weather buffer: add 12 h to the promised time on Storm / Fog days
    risky = df["weather"].isin(["Storm", "Fog"])
    still = (df["delivery_time_hrs"] > df["promised_time_hrs"] + 12)
    avoided_w = int(df.loc[risky, "delayed"].sum() - still[risky].sum())
    res["weather"] = {"risky_shipments": int(risky.sum()), "delays_before": int(df.loc[risky, "delayed"].sum()),
                      "avoided": avoided_w, "share_of_all": avoided_w / total_delays}
    # 3) mode shift: 30% of Standard road shipments >= 800 km move to rail
    band = df[(df["priority"] == "Standard") & (df["distance_km"] >= 800)]
    kgkm = band["weight_kg"] * band["distance_km"]
    ckk = band.groupby("transport_mode")["transport_cost_usd"].sum() / kgkm.groupby(band["transport_mode"]).sum()
    road = band[band["transport_mode"] == "Road"]
    saving = 0.30 * (ckk["Road"] - ckk["Rail"]) * (road["weight_kg"] * road["distance_km"]).sum()
    t = band.groupby("transport_mode")["delivery_time_hrs"].median()
    res["mode_shift"] = {"road_ckk": ckk["Road"], "rail_ckk": ckk["Rail"], "eligible": int(len(road)),
                         "saving": saving, "saving_pct": saving / df["transport_cost_usd"].sum(),
                         "road_time": t["Road"], "rail_time": t["Rail"]}
    # 4) peak-season capacity at Mumbai hub
    mum = df[df["origin_hub"] == "Mumbai"]
    res["mumbai"] = {"peak_rate": mum.loc[mum["is_peak"], "delayed"].mean(),
                     "off_rate": mum.loc[~mum["is_peak"], "delayed"].mean(),
                     "peak_shipments": int(mum["is_peak"].sum())}
    # [[/SCEN]]
    return res


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return to_py(o.to_dict(orient="index"))
    if isinstance(o, pd.Series):
        return to_py(o.to_dict())
    return o


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    raw_clean = simulate_dataset()
    raw = inject_data_issues(raw_clean)
    raw.to_csv(os.path.join(DATA_DIR, "logistics_shipments_raw.csv"), index=False)
    df, report = clean_dataset(raw)
    df.to_csv(os.path.join(DATA_DIR, "logistics_shipments_clean.csv"), index=False)

    R = {"cleaning": report}
    R["summary"] = summary_statistics(df)
    R["overall"] = {
        "shipments": len(df), "delay_rate": df["delayed"].mean(), "mean_time": df["delivery_time_hrs"].mean(),
        "mean_cost": df["transport_cost_usd"].mean(), "total_cost": df["transport_cost_usd"].sum(),
        "mean_delay_when_late": df.loc[df["delayed"] == 1, "delay_hours"].mean(),
        "mean_rating": df["customer_rating"].mean(),
        "rating_late": df.loc[df["delayed"] == 1, "customer_rating"].mean(),
        "rating_ontime": df.loc[df["delayed"] == 0, "customer_rating"].mean(),
    }
    R["by_mode"] = df.groupby("transport_mode").agg(
        shipments=("shipment_id", "count"), mean_time=("delivery_time_hrs", "mean"),
        median_time=("delivery_time_hrs", "median"), mean_cost=("transport_cost_usd", "mean"),
        cost_per_kg=("cost_per_kg", "median"), delay_rate=("delayed", "mean"),
        avg_distance=("distance_km", "mean")).reindex(MODES)
    R["by_carrier"] = df.groupby("carrier").agg(
        shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"),
        mean_time=("delivery_time_hrs", "mean"), mean_cost=("transport_cost_usd", "mean"),
        cost_per_kg=("cost_per_kg", "median"), rating=("customer_rating", "mean"))
    R["by_hub"] = df.groupby("origin_hub").agg(
        shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"),
        mean_time=("delivery_time_hrs", "mean"), mean_cost=("transport_cost_usd", "mean")).reindex(HUBS)
    R["by_weather"] = df.groupby("weather").agg(
        shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"),
        mean_delay=("delay_hours", "mean"))
    R["by_priority"] = df.groupby("priority").agg(
        shipments=("shipment_id", "count"), delay_rate=("delayed", "mean"),
        mean_time=("delivery_time_hrs", "mean"), mean_cost=("transport_cost_usd", "mean"))

    fig1_distributions(df)
    fig2_boxplots(df)
    corr = fig3_correlation(df)
    monthly = fig4_monthly_trend(df)
    fig5_distance_vs_time(df)
    pv = fig6_hub_carrier_heatmap(df)
    wp = fig7_weather_priority(df)
    cost_beta, cost_r2, time_beta, time_r2 = driver_models(df)
    fig8_drivers(cost_beta, cost_r2, time_beta, time_r2)
    lane, cum = fig9_pareto(df)

    R["slopes"] = {m: float(np.polyfit(g["distance_km"], g["delivery_time_hrs"], 1)[0] * 100)
                   for m, g in df.groupby("transport_mode")}            # extra hours per +100 km
    ex = df[(df["origin_hub"] == "Mumbai") & (df["destination_region"] == "South") & (df["carrier"] == "TransGo")
            & (df["transport_mode"] == "Road") & (df["delayed"] == 1) & df["customer_rating"].notna()].iloc[0]
    cell = df[(df["origin_hub"] == "Mumbai") & (df["carrier"] == "TransGo")]
    R["worked_row"] = ex.to_dict()
    R["worked_cell"] = {"n": len(cell), "late": int(cell["delayed"].sum())}
    R["corr"] = corr
    R["monthly"] = monthly
    R["hub_carrier"] = pv
    R["weather_priority"] = wp
    R["cost_beta"], R["cost_r2"] = cost_beta, cost_r2
    R["time_beta"], R["time_r2"] = time_beta, time_r2
    R["pareto"] = {"top10": lane.head(10), "cum10": cum.head(10),
                   "lanes_for_80pct": int((cum < 80).sum() + 1), "n_lanes": int(len(lane))}
    R["tests"] = {k: [float(x) for x in v] for k, v in run_tests(df).items()}
    R["scenarios"] = scenarios(df)
    hub_delay = df.groupby("origin_hub")["delay_hours"].sum()
    R["hub_delay_share"] = hub_delay / hub_delay.sum()
    R["peak"] = {"peak_delay": df.loc[df["is_peak"], "delayed"].mean(),
                 "off_delay": df.loc[~df["is_peak"], "delayed"].mean(),
                 "peak_cost": df.loc[df["is_peak"], "transport_cost_usd"].mean(),
                 "off_cost": df.loc[~df["is_peak"], "transport_cost_usd"].mean()}

    with open(os.path.join(BASE, "results.json"), "w") as f:
        json.dump(to_py(R), f, indent=2, default=str)
    print("Done. Shipments analysed:", len(df), "| overall delay rate: %.1f%%" % (100 * df["delayed"].mean()))


if __name__ == "__main__":
    main()
