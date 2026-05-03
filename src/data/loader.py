"""
Financial data loader using yfinance.

Pulls a company's balance sheet, income statement, and quarterly EPS history,
then computes a small set of ratios used elsewhere by the CSP solver and KB.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf


def _safe_get(series, key):
    """Return series[key] as float, or None if the row is missing/NaN."""
    if key not in series.index:
        return None
    value = series[key]
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    # yfinance uses NaN for missing periods
    return value if value == value else None


def get_financial_ratios(ticker: str) -> dict:
    """
    Fetch the most recent annual balance sheet + income statement for `ticker`
    and compute classical solvency / liquidity / profitability / valuation
    ratios.

    Returns:
        {
          "ticker": "AAPL",
          "debt_to_equity": float | None,
          "current_ratio": float | None,
          "interest_coverage_ratio": float | None,
          "pe_ratio": float | None,
          "roe": float | None,
          "gross_margin": float | None,
          "net_profit_margin": float | None,
          "quick_ratio": float | None,
        }
    """
    stock = yf.Ticker(ticker)

    # Both DataFrames: rows = line items, columns = reporting dates (newest first)
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt

    # Guard: yfinance returns an empty DataFrame when rate-limited or when a
    # ticker has no filing history. Indexing an empty frame raises IndexError,
    # so we return None-filled ratios early rather than crashing.
    if balance_sheet is None or balance_sheet.empty or balance_sheet.shape[1] == 0:
        try:
            info = stock.info or {}
        except Exception:
            info = {}
        return {
            "ticker": ticker.upper(),
            "debt_to_equity": None,
            "current_ratio": None,
            "interest_coverage_ratio": None,
            "pe_ratio": _coerce_float(info.get("trailingPE")),
            "roe": _coerce_float(info.get("returnOnEquity")),
            "gross_margin": _coerce_float(info.get("grossMargins")),
            "net_profit_margin": _coerce_float(info.get("profitMargins")),
            "quick_ratio": _coerce_float(info.get("quickRatio")),
        }

    # Pull the most recent fiscal year (column 0)
    bs = balance_sheet.iloc[:, 0]
    is_ = income_stmt.iloc[:, 0] if (income_stmt is not None and not income_stmt.empty and income_stmt.shape[1] > 0) else None

    # --- Debt-to-Equity = Total Debt / Stockholders Equity ---
    # Leverage gauge: how much of the firm is financed by creditors vs. owners.
    # A high D/E means more interest obligations and more risk in a downturn.
    total_debt = _safe_get(bs, "Total Debt")
    equity = _safe_get(bs, "Stockholders Equity")
    debt_to_equity = (
        total_debt / equity if total_debt is not None and equity not in (None, 0) else None
    )

    # --- Current Ratio = Current Assets / Current Liabilities ---
    # Short-term liquidity: can the firm cover obligations due within one year?
    # >1 means current assets exceed current liabilities (healthy short-term cushion).
    cur_assets = _safe_get(bs, "Current Assets")
    cur_liab = _safe_get(bs, "Current Liabilities")
    current_ratio = (
        cur_assets / cur_liab if cur_assets is not None and cur_liab not in (None, 0) else None
    )

    # --- Interest Coverage = EBIT / Interest Expense ---
    # Solvency gauge: how many times over operating earnings can cover interest.
    # Lenders typically want this >= 3. yfinance can sign Interest Expense
    # either way depending on the firm, so take the absolute value.
    ebit = None
    interest = None
    if is_ is not None:
        ebit = _safe_get(is_, "EBIT")
        if ebit is None:
            ebit = _safe_get(is_, "Operating Income")
        interest = _safe_get(is_, "Interest Expense")
        if interest is not None:
            interest = abs(interest)
    interest_coverage_ratio = (
        ebit / interest if ebit is not None and interest not in (None, 0) else None
    )

    # --- Valuation + profitability metrics from yfinance's `info` dict ---
    # `info` aggregates ratios yfinance has already computed against trailing
    # data, so we surface them directly rather than re-deriving from the
    # statements. Wrapped so a slow or failing `info` call doesn't break
    # the rest of the loader.
    try:
        info = stock.info or {}
    except Exception:
        info = {}

    pe_ratio = _coerce_float(info.get("trailingPE"))
    roe = _coerce_float(info.get("returnOnEquity"))
    gross_margin = _coerce_float(info.get("grossMargins"))
    net_profit_margin = _coerce_float(info.get("profitMargins"))
    quick_ratio = _coerce_float(info.get("quickRatio"))

    return {
        "ticker": ticker.upper(),
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "interest_coverage_ratio": interest_coverage_ratio,
        "pe_ratio": pe_ratio,
        "roe": roe,
        "gross_margin": gross_margin,
        "net_profit_margin": net_profit_margin,
        "quick_ratio": quick_ratio,
    }


def _coerce_float(value):
    """Cast to float, treating None/NaN/non-numeric as None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _to_naive_timestamp(value):
    """
    Coerce `value` to a tz-naive pandas Timestamp, or return None if it
    can't be parsed. We strip tz info so that comparisons across mixed
    sources (some tz-aware, some tz-naive) don't raise.
    """
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            try:
                ts = ts.tz_localize(None)
            except Exception:
                return None
    return ts


def _next_earnings_date(stock):
    """
    Return the next earnings date as 'May 15, 2026', or None.

    yfinance exposes upcoming earnings two ways and they disagree across
    versions, so we try both: `calendar` (which can be a dict or DataFrame
    depending on the release) and `earnings_dates` (a DataFrame indexed by
    timestamp). We normalize every candidate to a tz-naive Timestamp so
    that comparisons never raise, then pick the earliest future date.
    """
    candidates = []

    # ---- calendar -----------------------------------------------------
    try:
        cal = stock.calendar
    except Exception:
        cal = None
    if cal is not None:
        try:
            if isinstance(cal, dict):
                value = cal.get("Earnings Date")
                if value is not None:
                    items = value if isinstance(value, (list, tuple)) else [value]
                    candidates.extend(items)
            elif "Earnings Date" in getattr(cal, "index", []):
                row = cal.loc["Earnings Date"]
                candidates.extend(row.tolist())
        except Exception:
            pass

    # ---- earnings_dates ----------------------------------------------
    try:
        df = stock.earnings_dates
    except Exception:
        df = None
    if df is not None and not getattr(df, "empty", True):
        try:
            candidates.extend(list(df.index))
        except Exception:
            pass

    now = pd.Timestamp.now()  # tz-naive
    best = None
    for c in candidates:
        ts = _to_naive_timestamp(c)
        if ts is None or ts < now:
            continue
        if best is None or ts < best:
            best = ts

    if best is None:
        return None
    try:
        return best.strftime("%B %d, %Y")
    except Exception:
        return None


def get_realtime_price(ticker: str) -> dict:
    """
    Fetch live market data for `ticker` via yfinance.

    Pulls from `fast_info` (cheap, near-realtime) and falls back to `info`
    for fields fast_info doesn't expose. Any field that can't be retrieved
    is returned as None rather than raising.

    Returns:
        {
          "ticker": "AAPL",
          "timestamp": "May 03, 2026 04:45 PM EST",
          "current_price": float | None,
          "previous_close": float | None,
          "price_change": float | None,
          "price_change_percent": float | None,
          "open": float | None,
          "day_high": float | None,
          "day_low": float | None,
          "volume": int | None,
          "fifty_two_week_high": float | None,
          "fifty_two_week_low": float | None,
          "market_cap": float | None,
          "dividend_yield": float | None,   # already in percent units (0.39 -> "0.39%")
          "beta": float | None,
          "next_earnings_date": str | None,  # "May 15, 2026"
          "error": str | None,
        }
    """
    # Stamp the response with when we pulled it. Local TZ name (e.g. "EST",
    # "PDT") gives us the right wall-clock context without an extra dep.
    now = datetime.now().astimezone()
    tz_label = now.tzname() or "local"
    timestamp = now.strftime("%B %d, %Y %I:%M %p ") + tz_label

    result = {
        "ticker": ticker.upper(),
        "timestamp": timestamp,
        "current_price": None,
        "previous_close": None,
        "price_change": None,
        "price_change_percent": None,
        "open": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "market_cap": None,
        "dividend_yield": None,
        "beta": None,
        "next_earnings_date": None,
        "error": None,
    }

    try:
        stock = yf.Ticker(ticker)
        fast = {}
        try:
            fast = dict(stock.fast_info)
        except Exception:
            fast = {}

        result["current_price"] = _coerce_float(fast.get("last_price"))
        result["open"] = _coerce_float(fast.get("open"))
        result["day_high"] = _coerce_float(fast.get("day_high"))
        result["day_low"] = _coerce_float(fast.get("day_low"))
        result["fifty_two_week_high"] = _coerce_float(fast.get("year_high"))
        result["fifty_two_week_low"] = _coerce_float(fast.get("year_low"))
        result["market_cap"] = _coerce_float(fast.get("market_cap"))
        result["previous_close"] = _coerce_float(fast.get("previous_close"))

        volume = _coerce_float(fast.get("last_volume"))
        if volume is None:
            volume = _coerce_float(fast.get("regular_market_volume"))
        result["volume"] = int(volume) if volume is not None else None

        # Fall back to the slower `info` dict for anything fast_info missed,
        # plus the fields that only live on `info`.
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        fallback_keys = {
            "current_price": ("currentPrice", "regularMarketPrice"),
            "open": ("open", "regularMarketOpen"),
            "day_high": ("dayHigh", "regularMarketDayHigh"),
            "day_low": ("dayLow", "regularMarketDayLow"),
            "volume": ("volume", "regularMarketVolume"),
            "fifty_two_week_high": ("fiftyTwoWeekHigh",),
            "fifty_two_week_low": ("fiftyTwoWeekLow",),
            "market_cap": ("marketCap",),
            "previous_close": ("previousClose", "regularMarketPreviousClose"),
        }
        for field, src_keys in fallback_keys.items():
            if result[field] is not None:
                continue
            for src_key in src_keys:
                val = _coerce_float(info.get(src_key))
                if val is not None:
                    result[field] = int(val) if field == "volume" else val
                    break

        # Fields that only come from `info`.
        result["dividend_yield"] = _coerce_float(info.get("dividendYield"))
        result["beta"] = _coerce_float(info.get("beta"))

        # Derived: price change vs. previous close.
        cp = result["current_price"]
        pc = result["previous_close"]
        if cp is not None and pc is not None:
            change = cp - pc
            result["price_change"] = change
            result["price_change_percent"] = (change / pc * 100) if pc else None

        # Next earnings date — uses calendar / earnings_dates with fallback.
        result["next_earnings_date"] = _next_earnings_date(stock)
    except Exception as e:
        result["error"] = f"Couldn't fetch realtime price for {ticker}: {e}"

    return result


def get_earnings_history(ticker: str, num_quarters: int = 8) -> dict:
    """
    Fetch the last `num_quarters` quarters of reported EPS for `ticker`.

    Uses yfinance's `earnings_dates` table, which mixes future estimates
    (Reported EPS = NaN) and past actuals — we drop the NaN rows.

    Returns:
        {
          "ticker": "AAPL",
          "quarterly_eps": { "YYYY-MM-DD": float, ... }   # newest first
        }
    """
    stock = yf.Ticker(ticker)
    df = stock.earnings_dates

    if df is None or df.empty or "Reported EPS" not in df.columns:
        return {"ticker": ticker.upper(), "quarterly_eps": {}}

    # Keep only quarters that have actually reported, then take the most recent N
    reported = df["Reported EPS"].dropna().head(num_quarters)

    quarterly_eps = {
        date.strftime("%Y-%m-%d"): float(eps) for date, eps in reported.items()
    }

    return {
        "ticker": ticker.upper(),
        "quarterly_eps": quarterly_eps,
    }
