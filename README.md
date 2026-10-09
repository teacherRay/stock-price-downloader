# Stock Price Downloader & Investment Health Analyzer

A comprehensive financial research desktop application and CLI tool developed in Python and GTK 3. It combines fundamental company health analysis, multi-period audited filing trends, interactive technical charts, historical CSV downloading with indicator enrichment, and traceable AI research prompt generation.

---

## Key Features

### 1. Three Distinct Research Pillars
Every research conclusion is separated into clear, traceable assessments:
- **Pillar A: Company Financial Quality (Heuristic Score 0–100)**: Evaluates profitability, operating cash conversion, solvency, and revenue/earnings expansion. Zero points are awarded for missing data, and overall scores are withheld if data coverage is below 60%. Differentiates industrial balance sheets from fractional-reserve financial institutions and ETFs.
- **Pillar B: Valuation Evidence**: Decoupled from operational quality. Displays market capitalization, Trailing P/E, Forward P/E, Price-to-Book, and PEG ratios to evaluate market pricing relative to growth without assuming quality implies undervalued shares.
- **Pillar C: Technical Conditions**: Historical price action and momentum analysis based on principles from *Day Trading with ChatGPT* by Saskia Adler, explicitly separating crossover triggers from ongoing trend alignment states.

### 2. Embedded Interactive Charts (Matplotlib in GTK 3)
A dedicated interactive chart tab embedded natively into GTK 3, sharing the exact calculation snapshot used across the data table, reports, and CSV exports:
- **Candlestick & Line Views**: Color-coded Japanese candlesticks (teal `#26a69a` / red `#ef5350`) with accurate wicks and real candle bodies, or closing price line view with area shading.
- **Toggleable Overlays**: SMA-20, SMA-50 (enabled by default), SMA-200, EMA-12, EMA-26, and Bollinger Bands with shaded volatility bands.
- **Aligned Volume Panel**: Positioned directly beneath price, colored by candle direction.
- **Optional Oscillator Panels**:
  - **RSI (14)**: With reference lines at 30 and 70 and shaded normal range.
  - **MACD (12, 26, 9)**: MACD line, Signal line, and color-coded green/red histogram bars.
  - **Stochastic Oscillator (14, 3)**: %K and %D lines with reference lines at 20 and 80.
- **Navigation & Presets**: Zoom, pan, and home tools via Matplotlib toolbar, plus one-click range buttons: **1M**, **3M**, **6M**, **1Y**, and **All**.
- **Interactive Crosshair & Live Readout**: Real-time inspection banner displaying Date, Open, High, Low, Close, Volume, and all active indicator values on mouse hover.
- **Daily vs. Weekly Aggregation**: Switch between Daily bars and properly aggregated Weekly OHLCV bars (Monday Open, High/Low range, Friday Close, total Volume, and weekly indicator math).
- **Chart Export**: Save the high-resolution multi-panel chart directly as PNG (`💾 Export PNG`).

### 3. Multi-Period Annual Statement Trends
- Tracks 3–4 years of audited annual income statement history (Revenue, Gross Profit, Gross Margin, Operating Income, Operating Margin, Net Income, Net Margin).
- Evaluates operational expansion or margin compression across multiple periods instead of relying on a single snapshot.

### 4. Normalized Financial Units & Multi-Currency Support
- Identifies data provider (`Yahoo Finance`), retrieval timestamp, and reporting period (`TTM`, `MRQ`, or `Annual`).
- **Debt-to-Equity Normalization**: Provider percentages are scaled to decimal leverage ratios (`0.17x (16.97%)`) and scored strictly on the decimal ratio.
- **Dynamic Currency Formatting**: Monetary metrics display their actual reporting currencies (`$`, `€`, `£`, `¥`) rather than defaulting to `$`.

### 5. Historical CSV Downloader with Proper Indicator Warm-up
- Separate calculation history from export window: fetches $\ge 350$ bars to warm up SMA-200, EMA-26, and MACD chronologically before slicing to the requested period.
- Produces mathematically identical indicator values for identical dates regardless of ascending or descending sort order.
- Generates structured CSVs named `<Mon>-<Day>-<Year>_<Ticker>_<TradingDays>.csv` (e.g. `Oct-8-2026_NVDA_38.csv`).

### 6. Traceable AI / ChatGPT Research Prompts
- Encloses verified snapshot data within strict delimiter boundaries:
  ```text
  === VERIFIED DATA START ===
  ...
  === VERIFIED DATA END ===
  ```
- Guardrails instruct the AI to separate observed facts from interpretations, evaluate quality and valuation independently, avoid hallucinating unverified figures, and frame recommendations around a defined investment horizon.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 through 3.14)
- Linux (Ubuntu/Debian, Fedora, Arch) with GTK 3 installed

#### On Ubuntu / Debian:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

### Install Python Dependencies
```bash
git clone https://github.com/teacherRay/stock-price-downloader.git
cd stock-price-downloader
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the Desktop GUI
```bash
python3 stock_downloader_gui.py
```
1. Enter a ticker symbol (e.g., `NVDA`, `AAPL`, `MSFT`) or click a Quick preset.
2. Select the number of trading sessions (e.g., `38`, `60`, `252`).
3. Click **📥 Download & Analyze**.
4. Browse the tabs:
   - **🏢 Company Quality & Valuation**: Rubric audit, coverage %, valuation multiples, narrative findings.
   - **📈 Multi-Period Trends**: Multi-year annual filing history and margin trajectory.
   - **📉 Interactive Charts**: Candlestick/line chart, moving averages, volume, RSI, MACD, Stochastic, crosshair, range zoom.
   - **📊 Technical Analysis**: Technical stance, signals list, and comprehensive synthesis report.
   - **🤖 AI Research Prompts**: Copyable prompt templates for ChatGPT / Claude / Gemini.
   - **📋 Historical Data Table**: Enriched tabular view of prices and indicators.

### 2. Command-Line Interface (CLI)
```bash
# Basic download with technical indicators:
python3 download_stock_prices.py NVDA 38 --indicators

# Ascending chronological sort order:
python3 download_stock_prices.py AAPL 60 --sort asc --indicators

# Close-only export:
python3 download_stock_prices.py MSFT 20 --close-only

# View CLI options:
python3 download_stock_prices.py --help
```

---

## Running the Test Suite

Run the 10 offline unit tests (deterministic data with no network dependencies):
```bash
python3 -m unittest -v test_version4.py
```

The test suite validates:
- Indicator calculations, zero-handling, and warm-up (SMA, EMA, RSI, Bollinger, MACD, Stochastic).
- Ascending vs. descending sort order equivalence.
- Debt-to-Equity scaling normalization.
- Multi-currency symbol mapping (`EUR`, `GBP`, `USD`, `JPY`).
- Missing data scoring and 60% coverage threshold withholding.
- Non-operating asset customization (Banking balance sheets and ETFs).
- Crossover events vs. ongoing alignment states.
- Request generation token protection against delayed asynchronous responses.
- Partial failure decoupling (price charts usable when fundamentals fail).
- Weekly bar aggregation mathematics and weekly indicator calculation.
- Matplotlib chart series arrays and GTK 3 interaction.

---

## Project Structure

```text
stock-price-downloader/
├── download_stock_prices.py    # CLI downloader & warm-up buffer engine
├── stock_analysis.py           # Technical indicators, fundamentals & rubric engine
├── stock_downloader_gui.py     # GTK 3 desktop app with embedded Matplotlib charts
├── test_version4.py            # Offline unit test suite (10 tests)
├── stock-downloader.desktop    # Linux desktop launcher entry
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Documentation
```

---

## License

This project is licensed under the MIT License.
