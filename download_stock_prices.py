#!/usr/bin/env python3
"""
download_stock_prices.py

A tool to download historical stock prices for a specified ticker symbol
and period of trading days, formatted as CSV.

Filename format: <ExtractionDate>_<Ticker>_<TradingDays>[.csv]
Example: Oct-8-2026_NVDA_38.csv
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from stock_analysis import analyze_stock_data, fetch_company_fundamentals
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stock_analysis import analyze_stock_data, fetch_company_fundamentals


def get_extraction_date_str(dt: Optional[datetime.datetime] = None) -> str:
    """
    Format extraction date as <Mon>-<D>-<YYYY> (no leading zero on day),
    e.g., 'Oct-8-2026'.
    """
    if dt is None:
        dt = datetime.datetime.now()
    # %b gives abbreviated month name (e.g. Oct)
    month = dt.strftime("%b")
    day = dt.day  # integer, no leading zero
    year = dt.year
    return f"{month}-{day}-{year}"


def fetch_stock_chart(ticker: str, min_bars: int = 350) -> List[Dict]:
    """
    Fetch historical daily trading data for `ticker` from Yahoo Finance API.
    Returns a list of daily data dicts with keys:
    'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.

    Guarantees sufficient historical bars for warm-up of indicators (SMA-200, EMA-26, MACD).
    """
    # Calculate period window with warm-up buffer.
    # Stock markets trade ~252 days/year (~5 days/week).
    now_ts = int(time.time())
    buffer_days = max(int(min_bars * 2.5), min_bars + 60)
    p1 = max(0, now_ts - buffer_days * 86400)

    def _request_yahoo(p_start: int, p_end: int) -> List[Dict]:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
            f"?period1={p_start}&period2={p_end}&interval=1d&includePrePost=false"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8")
                data = json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(
                    f"Ticker '{ticker}' not found or no market data available."
                ) from e
            raise RuntimeError(
                f"Failed to fetch data from Yahoo Finance (HTTP {e.code}): {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Network error while connecting to Yahoo Finance: {e.reason}"
            ) from e

        chart = data.get("chart", {})
        error = chart.get("error")
        if error:
            desc = error.get("description", "Unknown error")
            raise ValueError(f"Yahoo Finance error for '{ticker}': {desc}")

        results = chart.get("result")
        if not results:
            raise ValueError(f"No price data returned for ticker '{ticker}'.")

        res = results[0]
        timestamps = res.get("timestamp", [])
        indicators = res.get("indicators", {})
        quotes = indicators.get("quote", [{}])[0]
        adjclose_list = (
            indicators.get("adjclose", [{}])[0].get("adjclose")
            if "adjclose" in indicators
            else None
        )

        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        volumes = quotes.get("volume", [])

        if adjclose_list is None:
            adjclose_list = closes

        records: List[Dict] = []
        for i, t in enumerate(timestamps):
            c = closes[i] if i < len(closes) else None
            # Skip invalid or market-closed/empty bars
            if c is None:
                continue

            o = opens[i] if i < len(opens) and opens[i] is not None else c
            h = highs[i] if i < len(highs) and highs[i] is not None else max(o, c)
            l = lows[i] if i < len(lows) and lows[i] is not None else min(o, c)
            v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
            ac = adjclose_list[i] if i < len(adjclose_list) and adjclose_list[i] is not None else c

            dt_utc = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
            date_str = dt_utc.strftime("%Y-%m-%d")

            records.append({
                "Date": date_str,
                "Open": round(float(o), 2),
                "High": round(float(h), 2),
                "Low": round(float(l), 2),
                "Close": round(float(c), 2),
                "Adj Close": round(float(ac), 2),
                "Volume": int(v),
            })
        return records

    records = _request_yahoo(p1, now_ts)

    # If the initial buffer wasn't enough trading days and p1 was > 0, fetch all history
    if len(records) < min_bars and p1 > 0:
        records = _request_yahoo(0, now_ts)

    if not records:
        raise ValueError(f"No trading records found for ticker '{ticker}'.")

    return records


def download_stock_prices(
    ticker: str,
    trading_days: int,
    output_dir: str = ".",
    sort_order: str = "desc",
    close_only: bool = False,
    include_no_ext_link: bool = True,
    include_indicators: bool = False,
    return_full_history: bool = False,
) -> Any:
    """
    Download historical stock prices and save to a CSV file.

    Parameters:
    -----------
    ticker : str
        Stock symbol (e.g. 'NVDA')
    trading_days : int
        Number of trading days requested
    output_dir : str
        Directory to write the CSV file to
    sort_order : str
        'desc' (newest first, default) or 'asc' (oldest first)
    close_only : bool
        If True, only write 'Date' and 'Close' columns
    include_no_ext_link : bool
        If True, also create a file/link without '.csv' extension to match
        both naming conventions.
    include_indicators : bool
        If True, calculate and append technical indicators from 'Day Trading with ChatGPT'
        (SMA, EMA, RSI, Bollinger Bands, MACD, Stochastic).

    Returns:
    --------
    Tuple[str, List[Dict]]: (output_csv_path, rows_written)
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        raise ValueError("Ticker symbol cannot be empty.")
    if trading_days <= 0:
        raise ValueError("Number of trading days must be a positive integer.")

    # Guarantee historical depth for indicator warm-up (SMA-200, EMA-26, MACD-9)
    min_calc_bars = max(trading_days + 250, 350)
    records = fetch_stock_chart(clean_ticker, min_calc_bars)

    # Chronologically compute all indicators on the full history
    analysis = analyze_stock_data(records, clean_ticker, calculation_history=records)
    enriched_records = analysis.get("enriched_records", records)

    total_available = len(enriched_records)
    if total_available < trading_days:
        print(
            f"Note: Only {total_available} trading days available for '{clean_ticker}' "
            f"(requested {trading_days}). Saving all {total_available} available trading days."
        )
        selected_records = list(enriched_records)
    else:
        # Take the most recent `trading_days` trading sessions
        selected_records = list(enriched_records[-trading_days:])

    # Sort order: ascending (chronological) vs descending (newest first)
    if sort_order == "desc":
        selected_records = list(reversed(selected_records))
    elif sort_order == "asc":
        pass
    else:
        raise ValueError("sort_order must be 'asc' or 'desc'.")

    # Generate filename: <Date>_<Ticker>_<TradingDays>.csv
    # e.g., Oct-8-2026_NVDA_38.csv
    date_str = get_extraction_date_str()
    base_name = f"{date_str}_{clean_ticker}_{trading_days}"
    csv_filename = f"{base_name}.csv"

    output_path = os.path.join(output_dir, csv_filename)

    if close_only:
        fieldnames = ["Date", "Close"]
    elif include_indicators:
        fieldnames = [
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14",
            "BB_Mid", "BB_Upper", "BB_Lower", "MACD", "MACD_Signal", "MACD_Hist",
            "Stoch_K", "Stoch_D"
        ]
    else:
        fieldnames = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Filter fields if close_only is specified
    if close_only:
        selected_records = [{k: r[k] for k in fieldnames} for r in selected_records]

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in selected_records:
            formatted_row = {
                k: f"{v:.2f}" if isinstance(v, float) else v
                for k, v in row.items()
            }
            writer.writerow(formatted_row)

    # Optional extensionless file / link for exact name match
    if include_no_ext_link:
        no_ext_path = os.path.join(output_dir, base_name)
        if os.path.abspath(output_path) != os.path.abspath(no_ext_path):
            try:
                if os.path.lexists(no_ext_path):
                    os.remove(no_ext_path)
                os.symlink(os.path.basename(output_path), no_ext_path)
            except (OSError, NotImplementedError):
                # Fallback to copy if symlink is not supported
                try:
                    import shutil
                    shutil.copyfile(output_path, no_ext_path)
                except Exception:
                    pass

    if return_full_history:
        return output_path, selected_records, list(enriched_records)
    return output_path, selected_records


def print_preview_table(rows: List[Dict], max_rows: int = 5):
    """Print a clean ASCII preview of the downloaded stock data."""
    if not rows:
        return
    keys = list(rows[0].keys())
    # Format header
    header = " | ".join(f"{k:>10}" for k in keys)
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    display_rows = rows if len(rows) <= max_rows * 2 else rows[:max_rows] + [None] + rows[-max_rows:]
    for r in display_rows:
        if r is None:
            print(" | ".join(f"{'...':>10}" for _ in keys))
        else:
            line = []
            for k in keys:
                val = r[k]
                if isinstance(val, float):
                    line.append(f"{val:>10.2f}")
                elif isinstance(val, int):
                    line.append(f"{val:>10,}")
                else:
                    line.append(f"{str(val):>10}")
            print(" | ".join(line))
    print(divider)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical stock prices for a ticker symbol and number of trading days."
    )
    parser.add_argument(
        "ticker_pos",
        nargs="?",
        metavar="TICKER",
        help="Stock ticker symbol (e.g. NVDA, AAPL, MSFT)",
    )
    parser.add_argument(
        "days_pos",
        nargs="?",
        type=int,
        metavar="DAYS",
        help="Number of trading days requested (e.g. 38)",
    )
    parser.add_argument(
        "-t", "--ticker",
        dest="ticker_flag",
        help="Stock ticker symbol",
    )
    parser.add_argument(
        "-d", "--days",
        dest="days_flag",
        type=int,
        help="Number of trading days",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--sort",
        choices=["desc", "asc"],
        default="desc",
        help="Sort order of dates: 'desc' (newest first, default) or 'asc' (oldest first)",
    )
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Include only Date and Close columns in the output CSV",
    )
    parser.add_argument(
        "--indicators",
        action="store_true",
        help="Include technical indicators from 'Day Trading with ChatGPT' as columns in the CSV",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip printing the technical analysis report",
    )

    args = parser.parse_args()

    # Determine ticker
    ticker = args.ticker_flag or args.ticker_pos
    days = args.days_flag if args.days_flag is not None else args.days_pos

    # If not provided via CLI, prompt interactively
    if not ticker:
        try:
            ticker = input("Enter stock ticker symbol (e.g., NVDA): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

    while not days:
        try:
            days_input = input("Enter number of trading days (e.g., 38): ").strip()
            days = int(days_input)
            if days <= 0:
                print("Error: Number of trading days must be greater than 0.")
                days = None
        except ValueError:
            print("Error: Please enter a valid integer.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(1)

    print(f"\nFetching {days} trading days for ticker '{ticker.upper()}'...")
    try:
        csv_file, rows = download_stock_prices(
            ticker=ticker,
            trading_days=days,
            output_dir=args.output_dir,
            sort_order=args.sort,
            close_only=args.close_only,
            include_indicators=args.indicators,
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully downloaded {len(rows)} trading days.")
    print(f"Data saved to: {csv_file}")
    print("\nPreview of extracted data:")
    print_preview_table(rows, max_rows=4)

    # Perform analysis separated into Three Distinct Pillars
    if not args.no_analyze and not args.close_only:
        clean_sym = ticker.strip().upper()
        # 1. Pillar A: Company Financial Quality & Pillar B: Valuation Evidence
        try:
            fund = fetch_company_fundamentals(clean_sym)
            if "error" not in fund:
                ev = fund.get("evaluation", {})
                print("\n" + "=" * 70)
                print("  PILLAR A: COMPANY FINANCIAL QUALITY (Heuristic Research Score)")
                print("=" * 70)
                print(f"Company: {fund.get('company_name')} ({fund.get('ticker')})")
                print(f"Sector: {fund.get('sector')} | Industry: {fund.get('industry')}")
                print(f"Data Provider: {fund.get('data_provider')} | Currency: {fund.get('currency')} | Retrieved: {fund.get('retrieval_timestamp')}")
                print(f"Data Coverage: {ev.get('coverage_pct', 0.0):.1f}% (Minimum threshold: 60.0%)")
                if ev.get("score_available"):
                    print(f"Quality Score: {ev.get('score')}/100 ({ev.get('rating')} - {ev.get('stance')})")
                else:
                    print(f"Quality Score: WITHHELD ({ev.get('rating')})")
                print(f"Assessment: {ev.get('verdict')}")
                if ev.get("caveat"):
                    print(f"Cautionary Note: {ev.get('caveat')}\n")

                print("KEY FINANCIAL METRICS (Audited / Provider Reported):")
                print(f"• Gross Profit Margin : {fund.get('gross_margin', {}).get('fmt'):<12} | Net Profit Margin   : {fund.get('net_margin', {}).get('fmt')}")
                print(f"• Return on Equity    : {fund.get('roe', {}).get('fmt'):<12} | Debt-to-Equity (D/E): {fund.get('debt_to_equity', {}).get('fmt')}")
                print(f"• Current Ratio       : {fund.get('current_ratio', {}).get('fmt'):<12} | Quick Ratio         : {fund.get('quick_ratio', {}).get('fmt')}")
                print(f"• Operating Cash Flow : {fund.get('operating_cashflow', {}).get('fmt'):<12} | Free Cash Flow (FCF): {fund.get('fcf', {}).get('fmt')}")
                print(f"• Revenue Growth (YoY): {fund.get('revenue_growth', {}).get('fmt'):<12} | Earnings Growth     : {fund.get('earnings_growth', {}).get('fmt')}")

                print("\nQUALITY HEURISTIC BREAKDOWN:")
                print(ev.get("detailed_analysis", ""))

                # 2. Pillar B: Valuation Evidence
                print("\n" + "=" * 70)
                print("  PILLAR B: VALUATION EVIDENCE (Decoupled from Quality)")
                print("=" * 70)
                print(f"• Market Capitalization : {fund.get('market_cap', {}).get('fmt')}")
                print(f"• Trailing P/E Ratio    : {fund.get('trailing_pe', {}).get('fmt')}")
                print(f"• Forward P/E Ratio     : {fund.get('forward_pe', {}).get('fmt')}")
                print(f"• Price-to-Book (P/B)   : {fund.get('price_to_book', {}).get('fmt')}")
                print(f"• PEG Ratio             : {fund.get('peg_ratio', {}).get('fmt')}")
                print("\nValuation Principle: High operational quality does not imply shares are cheap.")
                print("Valuation multiples must be assessed relative to growth and interest rates.")
            else:
                print(f"\n[Fundamentals Notice] {fund.get('error')}")
        except Exception as e:
            print(f"\n[Fundamentals Error] {e}")

        # 3. Pillar C: Technical Conditions
        print("\n" + "=" * 70)
        print("  PILLAR C: TECHNICAL CONDITIONS (from 'Day Trading with ChatGPT')")
        print("=" * 70)
        try:
            analysis = analyze_stock_data(rows, clean_sym)
            print(analysis.get("synthesis_report", "No report available."))
            print("=" * 70)
        except Exception as e:
            print(f"Technical Analysis error: {e}")


if __name__ == "__main__":
    main()
