"""
test_version4.py

Focused offline tests for Version 4 improvements:
1. Indicator calculations and warm-up policies (SMA, EMA, RSI, Bollinger, MACD, Stochastic).
2. Ascending vs Descending sort equivalence for historical indicators.
3. Missing-data scoring and minimum coverage threshold (withholding scores, no free points).
4. Debt-to-Equity scaling normalization and multi-currency formatting.
5. Stale-request token protection and partial failure handling (price succeeds while fundamentals fail).
6. Non-operating asset types (ETFs and Financial Institutions) handling.
"""

import math
import unittest
from typing import Any, Dict, List


class TestVersion4(unittest.TestCase):
    def setUp(self):
        # Deterministic price series for offline testing (50 sessions)
        self.prices = [
            100.0 + i * 1.5 + (5.0 if i % 2 == 0 else -5.0)
            for i in range(50)
        ]
        self.highs = [p + 2.0 for p in self.prices]
        self.lows = [p - 2.0 for p in self.prices]
        self.dates = [f"2026-01-{i+1:02d}" for i in range(50)]
        self.records = [
            {
                "Date": self.dates[i],
                "Open": self.prices[i] - 0.5,
                "High": self.highs[i],
                "Low": self.lows[i],
                "Close": self.prices[i],
                "Adj Close": self.prices[i],
                "Volume": 1000000 + i * 10000,
            }
            for i in range(50)
        ]

    def test_indicator_warmup_and_calculation(self):
        """Test indicator math, zero-handling, and warm-up handling."""
        from stock_analysis import (
            calculate_sma,
            calculate_ema,
            calculate_rsi,
            calculate_bollinger_bands,
            calculate_macd,
            calculate_stochastic,
        )

        # SMA
        sma_20 = calculate_sma(self.prices, 20)
        self.assertIsNone(sma_20[0])
        self.assertIsNone(sma_20[18])
        self.assertIsNotNone(sma_20[19])
        expected_sma = sum(self.prices[:20]) / 20.0
        self.assertAlmostEqual(sma_20[19], expected_sma, places=4)

        # EMA
        ema_12 = calculate_ema(self.prices, 12)
        self.assertIsNone(ema_12[0])
        self.assertIsNone(ema_12[10])
        self.assertIsNotNone(ema_12[11])
        # Initial EMA is SMA of first 12
        self.assertAlmostEqual(ema_12[11], sum(self.prices[:12]) / 12.0, places=4)
        # Next EMA matches formula
        k = 2.0 / 13.0
        expected_ema_12 = (self.prices[12] - ema_12[11]) * k + ema_12[11]
        self.assertAlmostEqual(ema_12[12], expected_ema_12, places=4)

        # RSI flat price edge case
        flat_prices = [100.0] * 30
        flat_rsi = calculate_rsi(flat_prices, 14)
        self.assertIsNotNone(flat_rsi[-1])
        self.assertEqual(flat_rsi[-1], 50.0)  # Neutral for zero change

        # Stochastic flat range edge case (High == Low)
        flat_highs = [100.0] * 20
        flat_lows = [100.0] * 20
        flat_closes = [100.0] * 20
        k_line, d_line = calculate_stochastic(flat_highs, flat_lows, flat_closes, 14, 3)
        self.assertIsNotNone(k_line[-1])
        self.assertEqual(k_line[-1], 50.0)  # Neutral instead of ZeroDivisionError

    def test_sort_order_equivalence(self):
        """Ensure ascending and descending exports produce identical indicator values for identical dates."""
        from stock_analysis import analyze_stock_data

        analysis_asc = analyze_stock_data(self.records, "TEST", calculation_history=self.records)
        enriched_asc = {r["Date"]: r for r in analysis_asc["enriched_records"]}

        records_desc = list(reversed(self.records))
        analysis_desc = analyze_stock_data(records_desc, "TEST", calculation_history=self.records)
        enriched_desc = {r["Date"]: r for r in analysis_desc["enriched_records"]}

        # Check that on the exact same date, indicators are identical
        test_date = self.dates[35]
        self.assertEqual(enriched_asc[test_date]["SMA_20"], enriched_desc[test_date]["SMA_20"])
        self.assertEqual(enriched_asc[test_date]["RSI_14"], enriched_desc[test_date]["RSI_14"])
        self.assertEqual(enriched_asc[test_date]["MACD"], enriched_desc[test_date]["MACD"])
        self.assertEqual(enriched_asc[test_date]["Stoch_K"], enriched_desc[test_date]["Stoch_K"])

    def test_debt_to_equity_scaling_and_normalization(self):
        """Verify Debt-to-Equity normalization from Yahoo percentage to decimal ratio."""
        from stock_analysis import _format_debt_to_equity

        # Case 1: Provider reports 16.971 (meaning 16.971% or 0.1697x)
        res1 = _format_debt_to_equity({"raw": 16.971, "fmt": "16.97%"})
        self.assertAlmostEqual(res1["ratio"], 0.16971, places=4)
        self.assertEqual(res1["pct_fmt"], "16.97%")
        self.assertIn("0.17x", res1["fmt"])

        # Case 2: High debt company: raw = 250.0 (250% = 2.50x)
        res2 = _format_debt_to_equity({"raw": 250.0, "fmt": "250.00%"})
        self.assertAlmostEqual(res2["ratio"], 2.50, places=2)
        self.assertIn("2.50x", res2["fmt"])

    def test_multi_currency_formatting(self):
        """Ensure currency formatting uses reporting currency (EUR, GBP, JPY, USD) without forcing '$'."""
        from stock_analysis import _format_metric

        eur_metric = _format_metric({"raw": 1500000000, "fmt": "1.50B"}, currency="EUR", is_currency=True)
        self.assertIn("€", eur_metric["fmt"])
        self.assertNotIn("$", eur_metric["fmt"])

        gbp_metric = _format_metric({"raw": 2400000000, "fmt": "2.40B"}, currency="GBP", is_currency=True)
        self.assertIn("£", gbp_metric["fmt"])
        self.assertNotIn("$", gbp_metric["fmt"])

        usd_metric = _format_metric({"raw": 500000000, "fmt": "500M"}, currency="USD", is_currency=True)
        self.assertIn("$", usd_metric["fmt"])

    def test_missing_data_scoring_and_coverage_threshold(self):
        """Verify that missing data awards 0 points and withholds score below 60% coverage."""
        from stock_analysis import evaluate_company_investment_worth

        # Sparse fundamentals mock with only 2 available metrics (< 60% coverage)
        sparse_fundamentals = {
            "ticker": "SPARSE",
            "company_name": "Sparse Corp",
            "sector": "Technology",
            "industry": "Software",
            "currency": "USD",
            "gross_margin": {"raw": 0.40, "fmt": "40.00%"},
            "net_margin": {"raw": None, "fmt": "N/A"},
            "roe": {"raw": None, "fmt": "N/A"},
            "current_ratio": {"raw": None, "fmt": "N/A"},
            "quick_ratio": {"raw": None, "fmt": "N/A"},
            "debt_to_equity": {"raw": None, "ratio": None, "fmt": "N/A"},
            "fcf": {"raw": None, "fmt": "N/A"},
            "operating_cashflow": {"raw": None, "fmt": "N/A"},
            "revenue_growth": {"raw": 0.10, "fmt": "10.00%"},
            "earnings_growth": {"raw": None, "fmt": "N/A"},
        }
        eval_sparse = evaluate_company_investment_worth(sparse_fundamentals)
        self.assertTrue(eval_sparse["score_withheld"])
        self.assertIn("Score Withheld", eval_sparse["rating"])
        self.assertLess(eval_sparse["coverage_pct"], 60.0)

    def test_non_operating_assets_etf_and_financials(self):
        """Verify that ETFs and Financial Services have customized handling and explain limitations."""
        from stock_analysis import evaluate_company_investment_worth

        etf_fundamentals = {
            "ticker": "SPY",
            "company_name": "SPDR S&P 500 ETF Trust",
            "quote_type": "ETF",
            "sector": "N/A",
            "industry": "N/A",
            "currency": "USD",
        }
        eval_etf = evaluate_company_investment_worth(etf_fundamentals)
        self.assertTrue(eval_etf["is_fund_or_etf"])
        self.assertTrue(eval_etf["score_withheld"])
        self.assertIn("ETF", eval_etf["rating"])

        bank_fundamentals = {
            "ticker": "JPM",
            "company_name": "JPMorgan Chase & Co.",
            "quote_type": "EQUITY",
            "sector": "Financial Services",
            "industry": "Banks - Diversified",
            "currency": "USD",
            "gross_margin": {"raw": None, "fmt": "N/A"},
            "net_margin": {"raw": 0.35, "fmt": "35.00%"},
            "roe": {"raw": 0.17, "fmt": "17.00%"},
            "current_ratio": {"raw": None, "fmt": "N/A"},
            "quick_ratio": {"raw": None, "fmt": "N/A"},
            "debt_to_equity": {"raw": None, "ratio": None, "fmt": "N/A"},
            "fcf": {"raw": None, "fmt": "N/A"},
            "operating_cashflow": {"raw": None, "fmt": "N/A"},
            "revenue_growth": {"raw": 0.08, "fmt": "8.00%"},
            "earnings_growth": {"raw": 0.12, "fmt": "12.00%"},
            "trailing_pe": {"raw": 12.5, "fmt": "12.50"},
        }
        eval_bank = evaluate_company_investment_worth(bank_fundamentals)
        self.assertTrue(eval_bank["is_financial_institution"])
        self.assertIn("Banking", eval_bank["detailed_analysis"])

    def test_crossover_event_vs_state(self):
        """Test that line above line is an alignment state, while crossing over between t-1 and t is an event."""
        from stock_analysis import detect_crossover_event

        # Series where A is above B on both days -> alignment, not a crossover event
        a_series_aligned = [10.0, 11.0]
        b_series_aligned = [8.0, 9.0]
        event_aligned = detect_crossover_event(a_series_aligned, b_series_aligned)
        self.assertFalse(event_aligned["crossed_above"])
        self.assertFalse(event_aligned["crossed_below"])
        self.assertTrue(event_aligned["is_above"])

        # Series where A was below B on day t-1 and above on day t -> crossover event
        a_series_cross = [8.0, 11.0]
        b_series_cross = [9.0, 10.0]
        event_cross = detect_crossover_event(a_series_cross, b_series_cross)
        self.assertTrue(event_cross["crossed_above"])
        self.assertTrue(event_cross["is_above"])

    def test_stale_request_token_and_partial_failures(self):
        """Verify that stale in-flight responses are dropped and partial failures leave prices usable."""
        from stock_downloader_gui import StockDownloaderApp
        from stock_analysis import analyze_stock_data

        app = StockDownloaderApp()

        # 1. Verify clean initial state: No analysis loaded, no 98/100 score
        self.assertIn("No analysis loaded", app.lbl_fund_company.get_text())
        self.assertIn("No Data", app.lbl_fund_badge.get_text())
        self.assertEqual(app.active_request_token, 0)

        # 2. Simulate User Request 1 for TICKER1 (token=1)
        app.active_request_token = 1
        app.current_ticker = "TICKER1"

        # User rapidly triggers Request 2 for TICKER2 (token=2)
        app.active_request_token = 2
        app.current_ticker = "TICKER2"

        # Stale response for TICKER1 arrives with token=1
        stale_analysis = analyze_stock_data(self.records, "TICKER1")
        ret1 = app._on_download_complete(
            token=1,
            ticker="TICKER1",
            csv_path="/tmp/TICKER1.csv",
            rows=self.records,
            full_records=self.records,
            analysis=stale_analysis,
            fund={"company_name": "Old Corp", "ticker": "TICKER1"},
            price_error=None,
            fund_error=None,
            include_indicators=True,
        )
        self.assertFalse(ret1)
        # Verify state was NOT overwritten by stale token 1
        self.assertNotEqual(app.last_saved_file, "/tmp/TICKER1.csv")

        # 3. Simulate partial failure: Token 2 has price data, but fundamentals failed
        analysis2 = analyze_stock_data(self.records, "TICKER2")
        app._on_download_complete(
            token=2,
            ticker="TICKER2",
            csv_path="/tmp/TICKER2.csv",
            rows=self.records,
            full_records=self.records,
            analysis=analysis2,
            fund=None,
            price_error=None,
            fund_error="HTTP 404 Not Found from provider",
            include_indicators=True,
        )

        # Price data and technicals are intact and usable
        self.assertEqual(app.last_saved_file, "/tmp/TICKER2.csv")
        self.assertEqual(len(app.current_rows), len(self.records))
        self.assertTrue(app.btn_open_file.get_sensitive())
        # Fundamentals section clearly reflects unavailable state, not old ticker
        self.assertIn("TICKER2", app.lbl_fund_company.get_text())
        self.assertIn("No Fundamentals", app.lbl_fund_badge.get_text())
        self.assertIn("WITHHELD", app.lbl_pillar_a_score.get_text())

    def test_weekly_aggregation_and_indicators(self):
        """Verify that daily records are accurately aggregated into weekly OHLCV bars and weekly indicators calculated."""
        from stock_analysis import aggregate_to_weekly, analyze_stock_data

        daily_data = [
            {"Date": "2026-01-05", "Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 104.0, "Volume": 1000},
            {"Date": "2026-01-06", "Open": 104.0, "High": 108.0, "Low": 103.0, "Close": 106.0, "Volume": 1200},
            {"Date": "2026-01-07", "Open": 106.0, "High": 107.0, "Low": 101.0, "Close": 102.0, "Volume": 800},
            {"Date": "2026-01-08", "Open": 102.0, "High": 110.0, "Low": 101.5, "Close": 109.0, "Volume": 1500},
            {"Date": "2026-01-09", "Open": 109.0, "High": 111.0, "Low": 108.0, "Close": 110.5, "Volume": 2000},
            # Next week
            {"Date": "2026-01-12", "Open": 111.0, "High": 115.0, "Low": 110.0, "Close": 114.0, "Volume": 1800},
            {"Date": "2026-01-13", "Open": 114.0, "High": 116.0, "Low": 113.0, "Close": 115.0, "Volume": 1400},
        ]

        weekly = aggregate_to_weekly(daily_data)
        self.assertEqual(len(weekly), 2)

        # Week 1 checks
        w1 = weekly[0]
        self.assertEqual(w1["Date"], "2026-01-09")  # Last day of week 1
        self.assertEqual(w1["Open"], 100.0)        # First Open of week 1
        self.assertEqual(w1["High"], 111.0)        # Max High of week 1
        self.assertEqual(w1["Low"], 99.0)          # Min Low of week 1
        self.assertEqual(w1["Close"], 110.5)       # Last Close of week 1
        self.assertEqual(w1["Volume"], 6500)       # Sum of volumes (1000+1200+800+1500+2000)

        # Week 2 checks
        w2 = weekly[1]
        self.assertEqual(w2["Date"], "2026-01-13")
        self.assertEqual(w2["Open"], 111.0)
        self.assertEqual(w2["High"], 116.0)
        self.assertEqual(w2["Low"], 110.0)
        self.assertEqual(w2["Close"], 115.0)
        self.assertEqual(w2["Volume"], 3200)

    def test_chart_series_structure_and_interaction(self):
        """Verify that analyze_stock_data returns series arrays for Matplotlib and app renders chart."""
        from stock_analysis import analyze_stock_data
        from stock_downloader_gui import StockDownloaderApp

        analysis = analyze_stock_data(self.records, "CHART_TEST", calculation_history=self.records)
        self.assertIn("series", analysis)
        series = analysis["series"]
        self.assertEqual(len(series["dates"]), len(self.records))
        self.assertEqual(len(series["closes"]), len(self.records))
        self.assertEqual(len(series["opens"]), len(self.records))
        self.assertEqual(len(series["highs"]), len(self.records))
        self.assertEqual(len(series["lows"]), len(self.records))
        self.assertEqual(len(series["volumes"]), len(self.records))
        self.assertEqual(len(series["sma_20"]), len(self.records))
        self.assertEqual(len(series["rsi_14"]), len(self.records))

        # Test app chart integration
        app = StockDownloaderApp()
        app.current_ticker = "CHART_TEST"
        app.full_history_records = self.records
        app.current_analysis = analysis

        # Verify chart rendering without crashing
        app._render_chart()
        self.assertTrue(len(app.chart_axes) >= 2)  # Price and Volume default active
        self.assertEqual(app.combo_chart_interval.get_active_id(), "daily")

        # Range change
        app._on_range_clicked("1m")
        self.assertEqual(app.current_chart_range, "1m")


if __name__ == "__main__":
    unittest.main()

