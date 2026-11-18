"""Shared helpers for the notebooks: data, features, the matrix-pricing baseline and metrics.

Data: cache/bond_panel.parquet, built from LDP by build_bond_panel.py (or any parquet named by REV_PANEL).
The notebooks only ever read the train and valid partitions; test stays closed.

    load()              train and valid rows
    xy(df, target)      allowed features and the target
    make_preprocessor   impute (+ missing flags), optionally scale, encode categories
    matrix_pricing      the desk baseline: median spread of other issuers' bonds in the same cell, same date
    score / table       MAE, RMSE, R squared in bps, overall and by IG / HY
    save                write a notebook's key numbers to results/<name>.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
PANEL = Path(os.environ.get("REV_PANEL", HERE / "cache" / "bond_panel.parquet"))
RESULTS = HERE / "results"
CACHE = HERE / "cache"
PARTS = ["train", "valid"]  # never "test"
SEED = 42
TARGET = "spread_bps"

# maturity_bucket is left out on purpose: it repeats years_to_maturity, and its integer codes would sort
# alphabetically (1-3y, 10-20y, 20y+, 3-5y, ...). It is still used by the matrix-pricing baseline.
CATEGORICAL = ["grade_bucket", "debt_type", "sector"]
NUMERIC = [
    # bond
    "years_to_maturity", "coupon_pct", "is_144a",
    # trading
    "n_trades", "log_par_volume", "has_inst_trade", "trade_days_in_month", "log_par_volume_month", "n_bonds_issuer",
    # issuer stock and balance sheet
    "vol_60d", "vol_252d", "ret_252d", "mom_12_1", "beta_252d", "book_leverage", "log_total_assets",
    # market
    "ust_2y_pct", "ust_10y_pct", "slope_2s10s_bps", "hy_oas_bps", "bbb_oas_bps", "vix", "sofr_pct",
]
FEATURES = NUMERIC + CATEGORICAL


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and valid rows of the panel."""
    if not PANEL.exists():
        raise FileNotFoundError(f"{PANEL} not found: run build_bond_panel.py first")
    df = pd.read_parquet(PANEL)
    df = df[df["partition"].isin(PARTS)].copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    train = df[df["partition"] == "train"].reset_index(drop=True)
    valid = df[df["partition"] == "valid"].reset_index(drop=True)
    return train, valid


def xy(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Allowed features (booleans as 0/1) and the target; rows with a missing target are dropped."""
    data = df[df[target].notna()]
    X = data[FEATURES].copy()
    for col in X.columns:
        if X[col].dtype == bool or str(X[col].dtype) == "boolean":
            X[col] = X[col].astype(float)
    return X, data[target]


def make_preprocessor(scale: bool = True, one_hot: bool = True) -> ColumnTransformer:
    """Median-impute numbers (with missing flags), optionally scale; encode categories.

    Trees: scale=False, one_hot=False (integer codes). KNN and neural networks: scale=True.
    """
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    enc = (OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False) if one_hot
           else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    cat = [("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encode", enc)]
    return ColumnTransformer([("num", Pipeline(num), NUMERIC), ("cat", Pipeline(cat), CATEGORICAL)],
                             verbose_feature_names_out=False)


def sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Random rows for slow models (KNN, MLP). Fine for fitting, never for validation."""
    return df if len(df) <= n else df.sample(n, random_state=seed)


LEVELS = [["grade_bucket", "maturity_bucket", "sector"], ["grade_bucket", "maturity_bucket"], ["grade_bucket"]]


def matrix_pricing(df: pd.DataFrame, min_bonds: int = 20) -> pd.Series:
    """Desk baseline: on each date, the median spread of OTHER issuers' bonds in the same grade x maturity x sector
    cell; cells with fewer than `min_bonds` bonds fall back to grade x maturity, then grade, then all bonds."""
    out = pd.Series(np.nan, index=df.index)
    for _, day in df.groupby("snapshot_date"):
        for iss, rows in day.groupby("issuer_ticker"):
            peers = day[day["issuer_ticker"] != iss]
            pred = pd.Series(np.nan, index=rows.index)
            for keys in LEVELS:
                g = peers.groupby(keys)[TARGET].agg(["median", "size"])
                med = g.loc[g["size"] >= min_bonds, "median"].rename("m")
                hit = rows[keys].merge(med, left_on=keys, right_index=True, how="left")["m"]
                pred = pred.fillna(pd.Series(hit.to_numpy(), index=rows.index))
            out.loc[rows.index] = pred.fillna(peers[TARGET].median()).to_numpy()
    return out


def baseline_valid(valid: pd.DataFrame) -> pd.Series:
    """matrix_pricing on the valid rows, cached (it loops over dates and issuers)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "baseline_valid.parquet"
    keys = valid[["cusip_id", "snapshot_date"]].reset_index(drop=True)
    if path.exists():
        cached = pd.read_parquet(path)
        if cached[["cusip_id", "snapshot_date"]].equals(keys):
            return pd.Series(cached["pred"].to_numpy(), index=valid.index)
    b = matrix_pricing(valid)
    keys.assign(pred=b.to_numpy()).to_parquet(path)
    return b


def regression_metrics(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(yt) | np.isnan(yp))
    err = yp[m] - yt[m]
    ss_tot = float(np.sum((yt[m] - yt[m].mean()) ** 2))
    return {"n": int(m.sum()), "mae_bps": float(np.mean(np.abs(err))), "rmse_bps": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")}


def score(valid: pd.DataFrame, pred, name: str) -> dict:
    """MAE / RMSE / R squared in bps on valid, overall and by IG / HY."""
    pred = np.asarray(pred, dtype=float)
    out = {"model": name, **regression_metrics(valid[TARGET], pred)}
    for g in ["IG", "HY"]:
        m = (valid["grade_bucket"] == g).to_numpy()
        out[f"mae_{g}"] = regression_metrics(valid.loc[m, TARGET], pred[m])["mae_bps"]
    return out


def table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[["model", "mae_bps", "mae_IG", "mae_HY", "rmse_bps", "r2", "n"]].round(3)


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
