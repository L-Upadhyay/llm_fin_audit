"""
Financial data loader using yfinance.

Pulls a company's balance sheet, income statement, and quarterly EPS history,
then computes a small set of ratios used elsewhere by the CSP solver and KB.
"""

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

    # Pull the most recent fiscal year (column 0)
    bs = balance_sheet.iloc[:, 0]
    is_ = income_stmt.iloc[:, 0]

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


def get_realtime_price(ticker: str) -> dict:
    """
    Fetch live market data for `ticker` via yfinance.

    Pulls from `fast_info` (cheap, near-realtime) and falls back to `info`
    for fields fast_info doesn't expose. Any field that can't be retrieved
    is returned as None rather than raising.

    Returns:
        {
          "ticker": "AAPL",
          "current_price": float | None,
          "open": float | None,
          "day_high": float | None,
          "day_low": float | None,
          "volume": int | None,
          "fifty_two_week_high": float | None,
          "fifty_two_week_low": float | None,
          "market_cap": float | None,
          "error": str | None,
        }
    """
    result = {
        "ticker": ticker.upper(),
        "current_price": None,
        "open": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "market_cap": None,
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

        volume = _coerce_float(fast.get("last_volume"))
        if volume is None:
            volume = _coerce_float(fast.get("regular_market_volume"))
        result["volume"] = int(volume) if volume is not None else None

        # Fall back to the slower `info` dict for anything fast_info missed.
        missing = [k for k, v in result.items() if v is None and k not in ("ticker", "error")]
        if missing:
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
            }
            for field in missing:
                for src_key in fallback_keys.get(field, ()):
                    val = _coerce_float(info.get(src_key))
                    if val is not None:
                        result[field] = int(val) if field == "volume" else val
                        break
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
