"""Forecasting baseline + variance analysis for renewable generation.

Everything here is deterministic (fixed seeds) and runs in well under a second on the
bundled dataset, so the API can build it once at start-up.

Terminology
-----------
* ``forecast_mwh``  : the operational (day-ahead) forecast supplied in the dataset.
* ``actual_mwh``    : metered generation.
* ``model``         : OUR weather+outage regression baseline, trained to predict ``actual_mwh``.
* ``miss`` / ``err``: actual - operational forecast (positive = under-forecast).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = BASE / "data" / "generation_forecast.csv"

RAW = ["irradiance_wm2", "wind_speed_ms", "temperature_c", "outage_flag"]
REQUIRED = ["date", *RAW, "forecast_mwh", "actual_mwh"]
# Human-friendly driver names used in the variance analysis / agent.
DRIVERS = {
    "irradiance_wm2": "irradiance",
    "wind_speed_ms": "wind speed",
    "temperature_c": "temperature",
    "outage_flag": "outage",
}
UNITS = {
    "irradiance_wm2": "W/m²",
    "wind_speed_ms": "m/s",
    "temperature_c": "°C",
    "outage_flag": "flag",
}
SEED = 42


# --------------------------------------------------------------------------- data
def load_data(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load + validate the dataset and add an ``asset`` column.

    The bundled CSV has exactly two rows per date and no site identifier. We assume the two
    rows are two generating units (or sites) and label them "A" / "B" by order of appearance
    within each date. If an ``asset`` / ``asset_id`` column exists it is used instead.
    """
    path = Path(path or os.getenv("DATA_PATH") or DEFAULT_DATA)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in RAW[:3] + ["forecast_mwh", "actual_mwh"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Dataset contains missing values; impute or drop before use.")
    if not set(df["outage_flag"].unique()) <= {0, 1}:
        raise ValueError("outage_flag must be 0/1")
    if (df[["forecast_mwh", "actual_mwh", "irradiance_wm2", "wind_speed_ms"]] < 0).any().any():
        raise ValueError("Negative generation / irradiance / wind values found.")

    for col in ("asset", "asset_id"):
        if col in df.columns:
            df["asset"] = df[col].astype(str)
            break
    else:
        # Original file order is block-wise (30 days for unit A, then 30 for unit B);
        # cumcount per date reproduces that and is robust to re-sorting.
        df["asset"] = df.groupby("date").cumcount().map(lambda i: chr(ord("A") + i))

    df["outage_flag"] = df["outage_flag"].astype(int)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    df["row_id"] = df["date"].dt.strftime("%Y-%m-%d") + "|" + df["asset"]
    df["err"] = df["actual_mwh"] - df["forecast_mwh"]
    return df


# ------------------------------------------------------------------- feature sets
def features_raw(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW].astype(float).copy()


def features_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-flavoured features.

    * irradiance in kW/m²; wind speed and a capped squared term (power ~ v^3 below rated, flat above);
    * ``heat_derate``: PV modules lose output as cell temperature rises, so only the part of
      temperature above 25 °C is used;
    * outage flag, plus an interaction so an outage scales down the weather-driven output
      instead of just subtracting a constant.
    """
    f = pd.DataFrame(index=df.index)
    irr = df["irradiance_wm2"] / 1000.0
    wind = df["wind_speed_ms"]
    out = df["outage_flag"].astype(float)
    f["irradiance"] = irr
    f["wind"] = wind
    f["wind_sq_capped"] = np.minimum(wind, 12.0) ** 2 / 12.0
    f["heat_derate"] = np.clip(df["temperature_c"] - 25.0, 0, None)
    f["outage"] = out
    f["outage_x_weather"] = out * (irr * 50 + wind * 4)
    return f


def features_corrected(df: pd.DataFrame) -> pd.DataFrame:
    """Operational forecast + outage terms: learns a bias / outage correction on top of it.

    Only usable when the operational forecast is available as an input (it is, day-ahead).
    """
    f = pd.DataFrame(index=df.index)
    f["forecast"] = df["forecast_mwh"]
    f["outage"] = df["outage_flag"].astype(float)
    f["forecast_x_outage"] = df["forecast_mwh"] * f["outage"]
    return f


def make_models() -> dict[str, tuple]:
    """name -> (estimator factory, feature function or None for non-feature baselines)."""
    return {
        "naive_mean": (None, None),
        "ridge_raw": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), features_raw),
        "ridge_engineered": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), features_engineered),
        "ridge_forecast_corrected": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=0.1)), features_corrected),
        "gbr_engineered": (
            lambda: GradientBoostingRegressor(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED
            ),
            features_engineered,
        ),
    }


# ----------------------------------------------------------------------- metrics
def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = a - p
    ss_res, ss_tot = float((e**2).sum()), float(((a - a.mean()) ** 2).sum())
    return {
        "n": int(len(a)),
        "mae": round(float(np.abs(e).mean()), 3),
        "rmse": round(float(np.sqrt((e**2).mean())), 3),
        "wape_pct": round(float(np.abs(e).sum() / a.sum() * 100), 2),
        "bias": round(float(e.mean()), 3),  # + = under-forecast
        "r2": round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else None,
    }


# ------------------------------------------------------------------ walk-forward CV
def walk_forward(df: pd.DataFrame, min_train_dates: int = 15, step: int = 3) -> dict:
    """Expanding-window, date-blocked validation.

    All rows of a date go to the same side of the split (the two units share weather, so
    splitting a date would leak). Each fold trains on every earlier date and predicts the next
    ``step`` dates.
    """
    dates = np.sort(df["date"].unique())
    models = make_models()
    oof = {name: pd.Series(np.nan, index=df.index) for name in models}
    fold_rows = []
    start = min_train_dates
    while start < len(dates):
        tr_dates, te_dates = dates[:start], dates[start : start + step]
        tr = df[df["date"].isin(tr_dates)]
        te = df[df["date"].isin(te_dates)]
        for name, (factory, feat) in models.items():
            if factory is None:
                pred = np.full(len(te), tr["actual_mwh"].mean())
            else:
                m = factory().fit(feat(tr), tr["actual_mwh"])
                pred = m.predict(feat(te))
            oof[name].loc[te.index] = np.clip(pred, 0, None)
        fold_rows.append(
            {
                "train_days": int(start),
                "test_from": str(pd.Timestamp(te_dates[0]).date()),
                "test_to": str(pd.Timestamp(te_dates[-1]).date()),
                **{f"mae_{n}": round(float(np.abs(te["actual_mwh"] - oof[n].loc[te.index]).mean()), 2) for n in models},
                "mae_operational": round(float(np.abs(te["err"]).mean()), 2),
            }
        )
        start += step

    scored = df[oof["naive_mean"].notna()]
    results = {"operational_forecast": metrics(scored["actual_mwh"], scored["forecast_mwh"])}
    for name in models:
        results[name] = metrics(scored["actual_mwh"], oof[name].loc[scored.index])

    # Same models, split by outage / normal days (do we handle outages better?)
    split = {}
    for label, mask in (("normal_days", scored["outage_flag"] == 0), ("outage_days", scored["outage_flag"] == 1)):
        sub = scored[mask]
        if len(sub):
            split[label] = {
                "operational_forecast": metrics(sub["actual_mwh"], sub["forecast_mwh"]),
                **{n: metrics(sub["actual_mwh"], oof[n].loc[sub.index]) for n in models if n != "naive_mean"},
            }
    return {"oof": oof, "scored_index": scored.index, "results": results, "folds": fold_rows, "split": split}


# --------------------------------------------------------------- variance analysis
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


@dataclass
class DriverModel:
    """Linear model of the forecast miss on the four raw drivers.

    Because actual and forecast are both (approximately) linear in the drivers, regressing
    ``actual - forecast`` on the drivers gives, for each driver, *how much the operational
    forecast mis-weights it*  (coefficient = beta_actual - beta_forecast). Per-row
    contribution = coefficient * (driver - fleet mean). What is left is the unexplained
    residual (forecast noise we cannot attribute).
    """

    coef: np.ndarray  # [intercept, b_irr, b_wind, b_temp, b_outage]
    means: np.ndarray
    ci: dict = field(default_factory=dict)
    r2: float = 0.0
    n_boot_valid: int = 0

    @classmethod
    def fit(cls, df: pd.DataFrame, n_boot: int = 500) -> "DriverModel":
        X, y = df[RAW].to_numpy(float), df["err"].to_numpy(float)
        coef = _ols(X, y)
        resid = y - np.column_stack([np.ones(len(X)), X]) @ coef
        r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        # Date-cluster bootstrap CI (the two units on one date share weather).
        rng = np.random.default_rng(SEED)
        dates = df["date"].unique()
        groups = {d: np.where(df["date"].to_numpy() == d)[0] for d in dates}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(len(dates), len(dates), replace=True)
            idx = np.concatenate([groups[dates[i]] for i in pick])
            if X[idx, 3].sum() == 0:  # resample without any outage row: coefficient undefined
                continue
            try:
                boots.append(_ols(X[idx], y[idx])[1:])
            except np.linalg.LinAlgError:
                continue
        boots = np.array(boots)
        n_valid = len(boots)
        lo, hi = np.percentile(boots, [5, 95], axis=0)
        ci = {RAW[i]: (float(lo[i]), float(hi[i])) for i in range(4)}
        return cls(coef=coef, means=X.mean(axis=0), ci=ci, r2=float(r2), n_boot_valid=n_valid)

    def contributions(self, row: pd.Series) -> dict:
        x = row[RAW].to_numpy(float)
        contrib = self.coef[1:] * (x - self.means)
        out = {DRIVERS[c]: float(v) for c, v in zip(RAW, contrib)}
        out["systematic bias"] = float(self.coef[0] + (self.coef[1:] * self.means).sum())
        out["unexplained"] = float(row["err"] - sum(out.values()))
        return out

    def table(self, df: pd.DataFrame) -> list[dict]:
        rows = []
        abs_total = 0.0
        per = {}
        for i, c in enumerate(RAW):
            v = np.abs(self.coef[i + 1] * (df[c].to_numpy(float) - self.means[i])).mean()
            per[c] = float(v)
            abs_total += v
        for i, c in enumerate(RAW):
            lo, hi = self.ci[c]
            rows.append(
                {
                    "driver": DRIVERS[c],
                    "unit": UNITS[c],
                    "coef_mwh_per_unit": round(float(self.coef[i + 1]), 4),
                    "ci90": [round(lo, 4), round(hi, 4)],
                    "significant": bool(lo > 0 or hi < 0),
                    "mean_abs_contribution_mwh": round(per[c], 3),
                    "share_of_explained_pct": round(per[c] / abs_total * 100, 1) if abs_total else 0.0,
                }
            )
        return rows


def severity(df: pd.DataFrame) -> pd.Series:
    """Robust z-score of the miss, scaled on non-outage days only."""
    base = df.loc[df["outage_flag"] == 0, "err"]
    med = base.median()
    mad = (base - med).abs().median() * 1.4826 or 1.0
    z = (df["err"] - med) / mad
    return z


def sev_label(z: float) -> str:
    a = abs(z)
    return "major" if a >= 2 else "moderate" if a >= 1.2 else "minor"


# ------------------------------------------------------------------- orchestrator
class Pipeline:
    def __init__(self, path: str | os.PathLike | None = None):
        self.df = load_data(path)
        self.cv = walk_forward(self.df)
        res = self.cv["results"]
        # Weather-only candidates: usable without the operational forecast, i.e. a true independent baseline.
        candidates = {k: v for k, v in res.items() if k in ("ridge_raw", "ridge_engineered", "gbr_engineered")}
        self.best_name = min(candidates, key=lambda k: candidates[k]["mae"])
        factory, feat = make_models()[self.best_name]
        self.feat = feat
        self.model = factory().fit(feat(self.df), self.df["actual_mwh"])
        oof_best = self.cv["oof"][self.best_name]
        self.df["model_forecast_oof"] = oof_best
        resid = (self.df["actual_mwh"] - oof_best).dropna()
        self.interval = (float(resid.quantile(0.10)), float(resid.quantile(0.90)))
        self.driver_model = DriverModel.fit(self.df)
        self.df["z"] = severity(self.df)
        self.df["severity"] = self.df["z"].map(sev_label)
        self.mae_noise_threshold = float(self.df.loc[self.df["outage_flag"] == 0, "err"].abs().median())

    # --- forecasting API
    def predict(self, irradiance_wm2: float, wind_speed_ms: float, temperature_c: float, outage_flag: int) -> dict:
        row = pd.DataFrame(
            [{"irradiance_wm2": irradiance_wm2, "wind_speed_ms": wind_speed_ms, "temperature_c": temperature_c, "outage_flag": outage_flag}]
        )
        p = float(max(self.model.predict(self.feat(row))[0], 0.0))
        lo, hi = self.interval
        return {
            "model": self.best_name,
            "forecast_mwh": round(p, 2),
            "interval_80": [round(max(p + lo, 0), 2), round(p + hi, 2)],
            "note": "80% interval from walk-forward out-of-fold residual quantiles; indicative only (n is small).",
        }

    # --- reporting helpers
    def summary(self) -> dict:
        df = self.df
        return {
            "data": {
                "rows": int(len(df)),
                "dates": int(df["date"].nunique()),
                "assets": sorted(df["asset"].unique().tolist()),
                "from": str(df["date"].min().date()),
                "to": str(df["date"].max().date()),
                "outage_rows": int(df["outage_flag"].sum()),
            },
            "operational_forecast_full_sample": metrics(df["actual_mwh"], df["forecast_mwh"]),
            "walk_forward": self.cv["results"],
            "walk_forward_split": self.cv["split"],
            "folds": self.cv["folds"],
            "selected_model": self.best_name,
            "drivers": self.driver_model.table(df),
            "driver_model_r2": round(self.driver_model.r2, 3),
            "bootstrap_samples_used": self.driver_model.n_boot_valid,
            "interval_80": [round(self.interval[0], 2), round(self.interval[1], 2)],
            "noise_floor_mwh": round(self.mae_noise_threshold, 2),
        }

    def series(self, asset: str | None = None) -> list[dict]:
        d = self.df if asset in (None, "", "all") else self.df[self.df["asset"] == asset]
        out = []
        for r in d.itertuples():
            out.append(
                {
                    "date": r.date.strftime("%Y-%m-%d"),
                    "asset": r.asset,
                    "actual": r.actual_mwh,
                    "forecast": r.forecast_mwh,
                    "model": None if pd.isna(r.model_forecast_oof) else round(float(r.model_forecast_oof), 2),
                    "err": round(float(r.err), 2),
                    "outage": int(r.outage_flag),
                    "severity": r.severity,
                }
            )
        return out

    def row(self, date: str, asset: str) -> pd.Series | None:
        m = self.df[(self.df["row_id"] == f"{date}|{asset.upper()}")]
        return None if m.empty else m.iloc[0]
