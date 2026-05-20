"""yfinance wrapper. Returns a price-per-ticker dict for a given date.

Falls back to last close if intraday is unavailable. Caches in-process only.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_close_prices(tickers: list[str]) -> dict[str, float]:
    """Return latest available close price per ticker.

    Uses the last 5 trading days and grabs the most recent close. Works
    whether the market is currently open or closed. Falls back to a
    serial per-ticker fetch for any ticker the batch download missed.
    """
    out: dict[str, float] = {}
    end = datetime.utcnow() + timedelta(days=1)
    start = end - timedelta(days=8)
    data = yf.download(
        tickers=" ".join(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=False,
    )
    for t in tickers:
        try:
            series = data[t]["Close"].dropna() if len(tickers) > 1 else data["Close"].dropna()
            out[t] = float(series.iloc[-1])
        except Exception:
            pass
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            df = yf.Ticker(t).history(period="7d", auto_adjust=True)
            if not df.empty:
                out[t] = float(df["Close"].dropna().iloc[-1])
        except Exception as e:
            print(f"[prices] failed to fetch {t}: {e}")
    return out


def fetch_history(tickers: list[str], days: int = 90) -> pd.DataFrame:
    """Return a DataFrame of daily adjusted-close prices for the last `days` calendar days.

    Columns are tickers, index is trading date. Missing tickers are dropped.
    Used by quant/stats.py to estimate the rolling covariance matrix.
    """
    end = datetime.utcnow() + timedelta(days=1)
    start = end - timedelta(days=int(days * 1.6) + 10)  # buffer for weekends/holidays
    data = yf.download(
        tickers=" ".join(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=False,
    )
    frames = {}
    for t in tickers:
        try:
            if len(tickers) > 1:
                series = data[t]["Close"].dropna()
            else:
                series = data["Close"].dropna()
            if len(series) >= 2:
                frames[t] = series
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    return df.tail(days)
