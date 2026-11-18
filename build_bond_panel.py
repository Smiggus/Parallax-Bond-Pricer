"""Build the bond panel from LDP (read-only): one row per bond per month-end, saved to cache/bond_panel.parquet.

    python build_bond_panel.py

Recipe
- Bonds: TRACE daily aggregates (credit.trace_daily). A bond enters a month if it traded in the last 7 calendar days
  of that month. Its yield is the par-volume-weighted average of the daily volume-weighted yields over those days
  (one day's yield is noisy: odd-lot retail prints move it a lot).
- Kept: plain corporate bonds (sub-product CORP: no equity-linked notes, no church bonds, no agencies), fixed coupon,
  non-convertible, investment-grade (I) or high-yield (H) flag, 1 to 30 years left. The I/H flag is TRACE's latest
  flag, not the flag at the snapshot date, and H also covers some unrated bonds.
- Issuer link: TRACE's company symbol is not always the company's stock ticker (old or reused symbols, subsidiaries).
  Within a symbol, only bonds from its main issuer name are kept, and the symbol must match the stock's company name
  in LDP (fundamentals.company_profile). Bonds that fail are dropped.
- Target: spread_bps = bond yield - Treasury par yield interpolated at the bond's remaining maturity, x100, kept
  between -50 and 2,500 bp. TRACE yields of callable bonds are yields to worst.
- Features: bond terms, trading activity, issuer stock (volatility, return, beta to SPY; at least 40 / 200 daily
  returns), balance sheet (book leverage, size) from the last filing known before the snapshot (filing date, or
  period end + 90 days; at most 400 days old), sector, and the market (Treasuries, credit index spreads, VIX, SOFR).
  Stock and market features are taken on the snapshot day (the month's last Treasury curve day).
- Classification target: widens_25bp = 1 if the spread is 25 bp or more wider at the next month-end (consecutive
  months only; empty otherwise).
- Partition by month: train 2024-01..2025-05, gap 2025-06, valid 2025-07..2025-10, gap 2025-11, test 2025-12 onward
  (TRACE in LDP ends mid-March 2026, so test holds 2025-12..2026-02).

Reads the LDP connection from LDP_DB_URL in the environment, or from the .env file named by REV_ENV (default: .env
next to this file; see .env.example). Never writes to LDP.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent
OUT = HERE / "cache" / "bond_panel.parquet"
ENV = Path(os.environ.get("REV_ENV", HERE / ".env"))
START, END = "2024-01-01", "2026-03-31"
LAST_DAYS = 7
SPREAD_RANGE = (-50, 2500)
MIN_RET_60, MIN_RET_252 = 40, 200
UST_TENORS = {"1M": 1 / 12, "2M": 2 / 12, "3M": 0.25, "4M": 4 / 12, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5,
              "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}
MACRO = {"UST_PAR_2Y": "ust_2y_pct", "UST_PAR_10Y": "ust_10y_pct", "VIXCLS": "vix", "BAMLH0A0HYM2": "hy_oas_pct",
         "BAMLC0A4CBBB": "bbb_oas_pct", "NYFED_SOFR": "sofr_pct"}
MATURITY_BINS = [1, 3, 5, 7, 10, 20, 30.01]
MATURITY_LABELS = ["1-3y", "3-5y", "5-7y", "7-10y", "10-20y", "20y+"]
NAME_STOP = {"THE", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LP", "LTD", "PLC", "SA", "NV", "AG", "HLDGS",
             "HOLDINGS", "HLDG", "GROUP", "GRP", "INTL", "INTERNATIONAL", "FIN", "FINL", "FINANCE", "FINANCIAL", "CAP",
             "CAPITAL", "FDG", "FUNDING", "NEW", "DEL", "OF", "AND", "&", "TR", "TRUST", "CR", "CREDIT", "MTN", "SR",
             "NT", "NTS", "USA", "US"}


def engine():
    url = os.environ.get("LDP_DB_URL") or dotenv_values(ENV).get("LDP_DB_URL")
    if not url:
        raise RuntimeError(f"LDP_DB_URL not set (environment or {ENV})")
    return create_engine(url, connect_args={"connect_timeout": 15})


def query(conn, sql, **params):
    return pd.read_sql(text(sql), conn, params=params)


def bond_months(conn) -> pd.DataFrame:
    """One row per bond and month: volume-weighted yield over its trading days in the month's last 7 days."""
    sql = """
    with d as (
        select cusip_id, trade_date, n_trades, par_volume, vw_yield, coalesce(n_trades_inst, 0) as n_trades_inst,
               date_trunc('month', trade_date)::date as mo
        from credit.trace_daily
        where trade_date between :start and :end and vw_yield > 0 and vw_yield < 40 and par_volume > 0
    ), month_all as (
        select cusip_id, mo, count(*) as trade_days_in_month, sum(par_volume) as par_volume_month
        from d group by cusip_id, mo
    ), last_week as (
        select cusip_id, mo, max(trade_date) as last_dt,
               sum(vw_yield * par_volume) / sum(par_volume) as yield_pct,
               sum(n_trades) as n_trades, sum(par_volume) as par_volume, sum(n_trades_inst) as n_trades_inst
        from d
        where trade_date >= (mo + interval '1 month' - interval '%d days')::date
        group by cusip_id, mo
    )
    select w.*, a.trade_days_in_month, a.par_volume_month
    from last_week w join month_all a using (cusip_id, mo)
    """ % LAST_DAYS
    return query(conn, sql, start=START, end=END)


def bond_reference(conn) -> pd.DataFrame:
    """Latest reference record per bond, filtered after the dedupe (an older passing record never stands in)."""
    sql = """
    select * from (
        select distinct on (cusip_id) cusip_id, company_symbol, issuer_nm, debt_type_cd, cpn_rt, mtrty_dt, ind_144a,
               grade, sub_prdct_type, cpn_type_cd, cnvrb_fl
        from credit.bond_reference
        order by cusip_id, enddt desc nulls first, fetched_at desc
    ) r
    where sub_prdct_type = 'CORP' and cpn_type_cd = 'FXPV' and cnvrb_fl = 'N' and grade in ('I', 'H')
      and company_symbol is not null
    """
    return query(conn, sql)


def macro(conn) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Daily macro series (forward-filled) and the days that have a published Treasury curve."""
    ids = list(MACRO) + [f"UST_PAR_{t}" for t in UST_TENORS]
    sql = "select series_id, date, value from macro.series_daily where series_id = any(:ids) and date between :s and :e"
    m = query(conn, sql, ids=list(set(ids)), s="2023-12-01", e=END)
    raw = m.pivot_table(index="date", columns="series_id", values="value").sort_index()
    raw.index = pd.to_datetime(raw.index)
    curve_days = raw.index[raw["UST_PAR_10Y"].notna()]
    return raw.ffill(), curve_days


def equities(conn, tickers) -> pd.DataFrame:
    sql = """
    select ticker, ts_event::date as date, adj_close
    from ohlcv.prices
    where resolution = 'daily' and ticker = any(:t) and ts_event >= :s and ts_event < :e and adj_close > 0
    """
    p = query(conn, sql, t=list(tickers) + ["SPY"], s="2022-12-01", e="2026-04-01")
    p["date"] = pd.to_datetime(p["date"])
    return p.drop_duplicates(["ticker", "date"]).pivot(index="date", columns="ticker", values="adj_close").sort_index()


def balance_sheets(conn, tickers) -> pd.DataFrame:
    sql = """
    select ticker, date as period_end, coalesce(filing_date, date + 90) as known_date, total_assets, total_debt
    from fundamentals.balance_sheet
    where ticker = any(:t) and total_assets > 0 and date >= '2022-06-01'
    """
    b = query(conn, sql, t=list(tickers))
    b["known_date"] = pd.to_datetime(b["known_date"])
    # one record per (ticker, known_date): the latest period end
    b = b.sort_values(["ticker", "known_date", "period_end"], kind="stable")
    return b.drop_duplicates(["ticker", "known_date"], keep="last").sort_values("known_date", kind="stable")


def profiles(conn, tickers) -> pd.DataFrame:
    sql = """
    select distinct on (ticker) ticker, company_name, sector
    from fundamentals.company_profile where ticker = any(:t)
    order by ticker, fetched_at desc
    """
    return query(conn, sql, t=list(tickers))


def tokens(name) -> list[str]:
    words = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    return [w for w in words if w not in NAME_STOP and not w.isdigit()]


def name_matches(issuer: str, company: str, symbol: str) -> bool:
    """Lenient check that a TRACE issuer name and a stock's company name are the same firm.

    True if an issuer word starts a company word (or the reverse) with at least 3 letters in common, or the issuer's
    first word equals the company's initials or the ticker (GS FIN CORP ~ Goldman Sachs Group, ticker GS).
    """
    it, ct = tokens(issuer), tokens(company)
    if not it or not ct:
        return False
    for a in it[:3]:
        for b in ct:
            k = min(len(a), len(b))
            if k >= 3 and (a.startswith(b) or b.startswith(a)):
                return True
    initials = "".join(w[0] for w in re.sub(r"[^A-Z ]", " ", str(company).upper()).split() if w not in NAME_STOP)
    return it[0] in {initials, symbol.upper()}


def clean_issuers(df: pd.DataFrame, prof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep each symbol's main issuer name, then only symbols whose stock company name matches it."""
    first = df["issuer_nm"].map(lambda n: (tokens(n) or ["?"])[0])
    main = (df.assign(first=first).drop_duplicates("cusip_id").groupby(["company_symbol", "first"]).size()
            .rename("bonds").reset_index().sort_values("bonds", ascending=False).drop_duplicates("company_symbol"))
    keep_first = dict(zip(main["company_symbol"], main["first"]))
    same_issuer = first.to_numpy() == df["company_symbol"].map(keep_first).to_numpy()
    out = df[same_issuer]
    names = out.drop_duplicates("company_symbol").set_index("company_symbol")["issuer_nm"]
    pmap = prof.set_index("ticker")["company_name"]
    ok = {s for s, n in names.items() if s in pmap.index and name_matches(n, pmap[s], s)}
    stats = {"rows_in": len(df), "dropped_other_issuer_in_symbol": int((~same_issuer).sum()),
             "symbols_checked": len(names), "symbols_name_mismatch": len(names) - len(ok)}
    out = out[out["company_symbol"].isin(ok)]
    stats["rows_out"] = len(out)
    return out, stats


def ust_at(curve_row: pd.Series, years: np.ndarray) -> np.ndarray:
    xs = np.array(list(UST_TENORS.values()))
    ys = np.array([curve_row.get(f"UST_PAR_{t}", np.nan) for t in UST_TENORS])
    ok = ~np.isnan(ys)
    return np.interp(years, xs[ok], ys[ok])


def equity_features(px: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Stock features per (ticker, snapshot) from daily adjusted closes up to the snapshot day."""
    ret = np.log(px).diff()
    out = []
    for d in dates:
        r = ret.loc[:d]
        if len(r) < 260:
            continue
        r60, r252 = r.iloc[-60:], r.iloc[-252:]
        spy = r252["SPY"]
        # beta on the days both the stock and SPY have a return (pairwise complete)
        n = r252.notna().mul(spy.notna(), axis=0).sum()
        beta = pd.Series({t: r252[t].cov(spy) / spy[r252[t].notna()].var() for t in r252.columns})
        f = pd.DataFrame({
            "vol_60d": (r60.std() * np.sqrt(252)).where(r60.count() >= MIN_RET_60),
            "vol_252d": (r252.std() * np.sqrt(252)).where(r252.count() >= MIN_RET_252),
            "ret_252d": np.exp(r252.sum(min_count=MIN_RET_252)) - 1,
            "mom_12_1": np.exp(r.iloc[-252:-21].sum(min_count=180)) - 1,
            "beta_252d": beta.where(n >= MIN_RET_252),
        })
        f["snapshot_date"] = d
        out.append(f.drop(index="SPY", errors="ignore").rename_axis("ticker").reset_index())
    return pd.concat(out, ignore_index=True)


def partition(d: pd.Series) -> pd.Series:
    out = pd.Series("gap", index=d.index, dtype=object)
    out[d <= "2025-05-31"] = "train"
    out[(d >= "2025-07-01") & (d <= "2025-10-31")] = "valid"
    out[(d >= "2025-12-01") & (d <= "2026-03-31")] = "test"
    return out


def main():
    eng = engine()
    with eng.connect() as conn:
        conn.execute(text("set transaction read only"))
        print("bond-months ...", flush=True)
        bm = bond_months(conn)
        ref = bond_reference(conn)
        df = bm.merge(ref, on="cusip_id", how="inner")
        print(f"  {len(bm):,} bond-months traded in a month's last week, {len(df):,} after the bond filters", flush=True)
        prof = profiles(conn, sorted(df["company_symbol"].unique()))
        df, link = clean_issuers(df, prof)
        print("  issuer link:", link, flush=True)
        tickers = sorted(df["company_symbol"].unique())
        print("macro, equities, balance sheets ...", flush=True)
        mac, curve_days = macro(conn)
        px = equities(conn, tickers)
        bs = balance_sheets(conn, tickers)

    # snapshot = the month's last day with a published Treasury curve
    df["mo"] = pd.to_datetime(df["mo"])
    df["last_dt"] = pd.to_datetime(df["last_dt"])
    snap = {mo: curve_days[(curve_days >= mo) & (curve_days < mo + pd.offsets.MonthBegin(1))].max()
            for mo in df["mo"].unique()}
    df["snapshot_date"] = df["mo"].map(snap)

    # bond terms and target (Treasury curve on the bond's last trading day)
    df["mtrty_dt"] = pd.to_datetime(df["mtrty_dt"])
    df["years_to_maturity"] = (df["mtrty_dt"] - df["last_dt"]).dt.days / 365.25
    df = df[(df["years_to_maturity"] >= 1) & (df["years_to_maturity"] <= 30)].reset_index(drop=True)
    ust = np.empty(len(df))
    for day, idx in df.groupby("last_dt").indices.items():
        ust[idx] = ust_at(mac.loc[:day].iloc[-1], df["years_to_maturity"].to_numpy()[idx])
    df["spread_bps"] = (df["yield_pct"] - ust) * 100
    df = df[df["spread_bps"].between(*SPREAD_RANGE)].copy()

    df["maturity_bucket"] = pd.cut(df["years_to_maturity"], MATURITY_BINS, labels=MATURITY_LABELS, right=False).astype(str)
    df["grade_bucket"] = df["grade"].map({"I": "IG", "H": "HY"})
    df["coupon_pct"] = df["cpn_rt"]
    df["is_144a"] = df["ind_144a"].eq("Y")
    df["debt_type"] = df["debt_type_cd"].fillna("Unknown")
    df["log_par_volume"] = np.log1p(df["par_volume"].clip(lower=0))
    df["log_par_volume_month"] = np.log1p(df["par_volume_month"].clip(lower=0))
    df["has_inst_trade"] = df["n_trades_inst"] > 0
    df["issuer_ticker"] = df["company_symbol"]

    # issuer: stock, balance sheet, sector
    eq = equity_features(px, sorted(df["snapshot_date"].unique()))
    df = df.merge(eq, left_on=["issuer_ticker", "snapshot_date"], right_on=["ticker", "snapshot_date"], how="left")
    df = df.sort_values("snapshot_date", kind="stable")
    bs = bs.rename(columns={"ticker": "issuer_ticker"})
    df = pd.merge_asof(df, bs, left_on="snapshot_date", right_on="known_date", by="issuer_ticker",
                       direction="backward", tolerance=pd.Timedelta("400D"))
    df["book_leverage"] = df["total_debt"] / df["total_assets"]
    df.loc[~df["book_leverage"].between(0, 1.5), "book_leverage"] = np.nan  # outside 0-150%: bad record
    df["log_total_assets"] = np.log(df["total_assets"])
    df = df.merge(prof[["ticker", "sector"]].rename(columns={"ticker": "issuer_ticker"}), on="issuer_ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")
    df["n_bonds_issuer"] = df.groupby(["issuer_ticker", "snapshot_date"])["cusip_id"].transform("count")

    # market on the snapshot day
    m = mac.reindex(sorted(df["snapshot_date"].unique()))[list(MACRO)].rename(columns=MACRO)
    m["slope_2s10s_bps"] = (m["ust_10y_pct"] - m["ust_2y_pct"]) * 100
    m["hy_oas_bps"] = m.pop("hy_oas_pct") * 100
    m["bbb_oas_bps"] = m.pop("bbb_oas_pct") * 100
    df = df.merge(m, left_on="snapshot_date", right_index=True, how="left")

    # next month's spread for the classification target (consecutive months only)
    df = df.sort_values(["cusip_id", "snapshot_date"])
    nxt = df.groupby("cusip_id")[["snapshot_date", "spread_bps"]].shift(-1)
    months_ahead = (nxt["snapshot_date"].dt.year - df["snapshot_date"].dt.year) * 12 + \
                   (nxt["snapshot_date"].dt.month - df["snapshot_date"].dt.month)
    df["spread_change_next_bps"] = (nxt["spread_bps"] - df["spread_bps"]).where(months_ahead == 1)
    df["widens_25bp"] = (df["spread_change_next_bps"] >= 25).astype(float).where(df["spread_change_next_bps"].notna())
    df["partition"] = partition(df["snapshot_date"])

    keep = ["snapshot_date", "cusip_id", "issuer_ticker", "issuer_nm", "partition", "spread_bps", "spread_change_next_bps",
            "widens_25bp", "years_to_maturity", "maturity_bucket", "coupon_pct", "grade_bucket", "is_144a", "debt_type",
            "n_trades", "log_par_volume", "has_inst_trade", "trade_days_in_month", "log_par_volume_month",
            "n_bonds_issuer", "sector", "vol_60d", "vol_252d", "ret_252d", "mom_12_1", "beta_252d", "book_leverage",
            "log_total_assets", "ust_2y_pct", "ust_10y_pct", "slope_2s10s_bps", "hy_oas_bps", "bbb_oas_bps", "vix",
            "sofr_pct"]
    df = df[keep].rename(columns={"issuer_nm": "issuer_name"}).sort_values(["snapshot_date", "cusip_id"])
    assert not df.duplicated(["cusip_id", "snapshot_date"]).any()
    OUT.parent.mkdir(exist_ok=True)
    df.reset_index(drop=True).to_parquet(OUT)

    print(f"wrote {OUT}: {len(df):,} rows, {df.cusip_id.nunique():,} bonds, {df.issuer_ticker.nunique():,} issuers")
    print(df.groupby("partition").agg(rows=("cusip_id", "size"), months=("snapshot_date", "nunique")).to_string())
    print(df.groupby("grade_bucket")["spread_bps"].describe().round(1).to_string())
    print("missing share:", df.isna().mean().round(3)[lambda s: s > 0].to_dict())
    print("sector Unknown share:", round(df["sector"].eq("Unknown").mean(), 3))
    w = df.dropna(subset=["spread_change_next_bps"]).sort_values(["cusip_id", "snapshot_date"])
    nxt2 = w.groupby("cusip_id")["spread_change_next_bps"].shift(-1)
    rev = ((w["widens_25bp"] == 1) & nxt2.notna())
    print("widens_25bp share:", round(df["widens_25bp"].mean(), 4),
          "| share of widenings that reverse by 25 bp next month:", round((nxt2[rev] <= -25).mean(), 3))


if __name__ == "__main__":
    main()
