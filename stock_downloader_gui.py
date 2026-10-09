#!/usr/bin/env python3
"""
stock_downloader_gui.py - Version 4.1 with Dedicated Interactive Charts

A comprehensive stock research and downloading application featuring:
1. Top Section: Three Distinct Research Pillars
   - Pillar A: Company Financial Quality (Heuristic Score, Minimum Coverage Threshold, Rubric Audit)
   - Pillar B: Valuation Evidence (Decoupled from Quality, Multiples vs Growth)
   - Pillar C: Technical Conditions (Attributed to 'Day Trading with ChatGPT' by Saskia Adler)
2. Four-column live grid of fundamental metrics with normalized units and currency mapping.
3. Multi-period annual income statement trend audit (3-4 years).
4. Dedicated Interactive Charts Tab embedded via Matplotlib in GTK 3:
   - Candlestick and Line view with SMA-20, SMA-50, SMA-200, EMA-12/26, Bollinger Bands.
   - Volume panel aligned beneath price chart.
   - Separate optional panels for RSI (with 30/70 lines), MACD, and Stochastic (with 20/80 lines).
   - Zoom, pan, reset view, range presets (1M, 3M, 6M, 1Y, All), and crosshair with OHLCV readout.
   - Daily and weekly aggregation views, and PNG export.
5. Delimited, anti-hallucination AI research prompts with strict investment horizon framing.
6. Historical CSV download with chronological indicator warm-up and identical asc/desc values.
7. Asynchronous execution with request generation tokens to eliminate stale-result overwrites.
"""

import datetime
import os
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk

# Matplotlib embedded in GTK 3
import matplotlib
matplotlib.use("GTK3Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3
import numpy as np

# Import download & analysis modules
try:
    from download_stock_prices import download_stock_prices, get_extraction_date_str
    from stock_analysis import (
        analyze_stock_data,
        fetch_company_fundamentals,
        evaluate_company_investment_worth,
        aggregate_to_weekly,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from download_stock_prices import download_stock_prices, get_extraction_date_str
    from stock_analysis import (
        analyze_stock_data,
        fetch_company_fundamentals,
        evaluate_company_investment_worth,
        aggregate_to_weekly,
    )


class StockDownloaderApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="Stock Price Downloader & Investment Health Analyzer v4.1")
        self.set_default_size(1080, 840)
        self.set_position(Gtk.WindowPosition.CENTER)

        # State management
        self.active_request_token: int = 0
        self.last_saved_file: Optional[str] = None
        self.current_ticker: Optional[str] = None
        self.current_analysis: Optional[Dict[str, Any]] = None
        self.current_fundamentals: Optional[Dict[str, Any]] = None
        self.current_rows: List[Dict] = []
        self.full_history_records: List[Dict] = []
        self.weekly_history_records: List[Dict] = []
        self.weekly_analysis: Optional[Dict[str, Any]] = None
        self.download_in_progress = False

        # Chart state
        self.current_chart_interval: str = "daily"
        self.current_chart_type: str = "candlestick"
        self.current_chart_range: str = "all"
        self.chart_active_series: Dict[str, Any] = {}
        self.ch_vline = None
        self.ch_hlines = []

        # Outer Scrolled Window
        outer_scroll = Gtk.ScrolledWindow()
        outer_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.add(outer_scroll)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.set_border_width(12)
        outer_scroll.add(main_box)

        # =========================================================
        # 1. Header Bar
        # =========================================================
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        title_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        lbl_title = Gtk.Label()
        lbl_title.set_markup("<span size='large' weight='bold'>📈 Stock Downloader &amp; Company Health Analyzer</span> <span size='small' color='#666666'>v4.1</span>")
        lbl_title.set_xalign(0)
        lbl_subtitle = Gtk.Label()
        lbl_subtitle.set_markup(
            "<span color='#555555'>Traceable research assessing Company Quality, Valuation, and Technical Conditions from <i>'Day Trading with ChatGPT'</i></span>"
        )
        lbl_subtitle.set_xalign(0)
        title_vbox.pack_start(lbl_title, False, False, 0)
        title_vbox.pack_start(lbl_subtitle, False, False, 0)
        header_box.pack_start(title_vbox, True, True, 0)
        main_box.pack_start(header_box, False, False, 0)

        main_box.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 2)

        # =========================================================
        # 2. Controls & Input Bar
        # =========================================================
        ctrl_grid = Gtk.Grid()
        ctrl_grid.set_column_spacing(10)
        ctrl_grid.set_row_spacing(6)
        main_box.pack_start(ctrl_grid, False, False, 0)

        # Ticker input
        lbl_ticker = Gtk.Label(label="Ticker:")
        lbl_ticker.set_xalign(0)
        ctrl_grid.attach(lbl_ticker, 0, 0, 1, 1)

        self.entry_ticker = Gtk.Entry()
        self.entry_ticker.set_placeholder_text("e.g. NVDA")
        self.entry_ticker.set_text("NVDA")
        self.entry_ticker.set_width_chars(8)
        self.entry_ticker.connect("activate", lambda w: self.on_download_clicked(None))
        ctrl_grid.attach(self.entry_ticker, 1, 0, 1, 1)

        quick_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        lbl_q = Gtk.Label(label="Quick:")
        lbl_q.get_style_context().add_class("dim-label")
        quick_box.pack_start(lbl_q, False, False, 0)
        for sym in ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]:
            btn = Gtk.Button(label=sym)
            btn.set_relief(Gtk.ReliefStyle.NONE)
            btn.connect("clicked", lambda b, s=sym: self._select_ticker(s))
            quick_box.pack_start(btn, False, False, 0)
        ctrl_grid.attach(quick_box, 2, 0, 1, 1)

        # Trading Days
        lbl_days = Gtk.Label(label="Days:")
        lbl_days.set_xalign(0)
        ctrl_grid.attach(lbl_days, 3, 0, 1, 1)

        adj_days = Gtk.Adjustment(value=38, lower=5, upper=10000, step_increment=1, page_increment=10)
        self.spin_days = Gtk.SpinButton(adjustment=adj_days, numeric=True)
        self.spin_days.set_width_chars(6)
        self.spin_days.connect("activate", lambda w: self.on_download_clicked(None))
        ctrl_grid.attach(self.spin_days, 4, 0, 1, 1)

        # Presets
        p_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        for d in [14, 20, 38, 60, 252]:
            btn = Gtk.Button(label=f"{d}d")
            btn.set_relief(Gtk.ReliefStyle.NONE)
            btn.connect("clicked", lambda b, val=d: self.spin_days.set_value(val))
            p_box.pack_start(btn, False, False, 0)
        ctrl_grid.attach(p_box, 5, 0, 1, 1)

        # Row 2: Folder, Sort, Indicator toggle
        lbl_fld = Gtk.Label(label="Folder:")
        lbl_fld.set_xalign(0)
        ctrl_grid.attach(lbl_fld, 0, 1, 1, 1)

        self.file_chooser = Gtk.FileChooserButton(title="Select Save Directory", action=Gtk.FileChooserAction.SELECT_FOLDER)
        self.file_chooser.set_current_folder(os.getcwd())
        ctrl_grid.attach(self.file_chooser, 1, 1, 2, 1)

        lbl_s = Gtk.Label(label="Sort:")
        lbl_s.set_xalign(0)
        ctrl_grid.attach(lbl_s, 3, 1, 1, 1)

        self.combo_sort = Gtk.ComboBoxText()
        self.combo_sort.append("desc", "Newest first (Desc)")
        self.combo_sort.append("asc", "Oldest first (Asc)")
        self.combo_sort.set_active_id("desc")
        ctrl_grid.attach(self.combo_sort, 4, 1, 1, 1)

        self.chk_include_indicators = Gtk.CheckButton(label="Enrich CSV with Indicators")
        self.chk_include_indicators.set_active(True)
        ctrl_grid.attach(self.chk_include_indicators, 5, 1, 1, 1)

        # Action Buttons Row
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        action_box.set_margin_top(2)
        action_box.set_margin_bottom(2)
        main_box.pack_start(action_box, False, False, 0)

        self.btn_download = Gtk.Button(label="📥 Download & Analyze")
        self.btn_download.get_style_context().add_class("suggested-action")
        self.btn_download.connect("clicked", self.on_download_clicked)
        action_box.pack_start(self.btn_download, False, False, 0)

        self.spinner = Gtk.Spinner()
        action_box.pack_start(self.spinner, False, False, 0)

        self.status_label = Gtk.Label()
        self.status_label.set_xalign(0)
        self.status_label.set_markup("<span color='#555555'>Ready. Enter ticker symbol and click Download &amp; Analyze.</span>")
        action_box.pack_start(self.status_label, True, True, 0)

        self.btn_open_file = Gtk.Button(label="Open CSV")
        self.btn_open_file.set_sensitive(False)
        self.btn_open_file.connect("clicked", self.on_open_file_clicked)
        action_box.pack_start(self.btn_open_file, False, False, 0)

        self.btn_open_folder = Gtk.Button(label="Show in Folder")
        self.btn_open_folder.set_sensitive(False)
        self.btn_open_folder.connect("clicked", self.on_open_folder_clicked)
        action_box.pack_start(self.btn_open_folder, False, False, 0)

        self.lbl_filename_preview = Gtk.Label()
        self.lbl_filename_preview.set_xalign(0)
        self.update_filename_preview()
        main_box.pack_start(self.lbl_filename_preview, False, False, 0)

        self.entry_ticker.connect("changed", lambda w: self.update_filename_preview())
        self.spin_days.connect("value-changed", lambda w: self.update_filename_preview())

        # =========================================================
        # 3. TOP SECTION: Three Distinct Research Pillars & Cards
        # =========================================================
        top_frame = Gtk.Frame(label=" 🏛️ Fundamental Company Health & Three Research Pillars ")
        main_box.pack_start(top_frame, False, False, 2)

        top_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        top_vbox.set_border_width(10)
        top_frame.add(top_vbox)

        # Banner line: Company name, Sector, Currency, Retrieval info
        banner_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        top_vbox.pack_start(banner_box, False, False, 0)

        self.lbl_fund_company = Gtk.Label()
        self.lbl_fund_company.set_markup("<b>No analysis loaded.</b> | Enter a ticker and click 'Download &amp; Analyze'")
        self.lbl_fund_company.set_xalign(0)
        banner_box.pack_start(self.lbl_fund_company, True, True, 0)

        self.lbl_fund_badge = Gtk.Label()
        self.lbl_fund_badge.set_markup("<span size='large' color='#888888'>⚪ No Data</span>")
        banner_box.pack_start(self.lbl_fund_badge, False, False, 0)

        btn_view_full_fund = Gtk.Button(label="🔍 View Rubric & Details")
        btn_view_full_fund.connect("clicked", lambda w: self.notebook.set_current_page(0))
        banner_box.pack_start(btn_view_full_fund, False, False, 0)

        # Three Distinct Summary Pillar Cards
        pillar_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        pillar_box.set_homogeneous(True)
        top_vbox.pack_start(pillar_box, False, False, 0)

        # Pillar A Card
        card_a = Gtk.Frame(label=" Pillar A: Company Financial Quality ")
        box_a = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box_a.set_border_width(6)
        self.lbl_pillar_a_score = Gtk.Label()
        self.lbl_pillar_a_score.set_markup("<span size='medium' weight='bold' color='#555555'>Score: —</span>")
        self.lbl_pillar_a_sub = Gtk.Label()
        self.lbl_pillar_a_sub.set_markup("<span size='small' color='#666666'>Coverage: — (Min 60%)</span>")
        box_a.pack_start(self.lbl_pillar_a_score, False, False, 0)
        box_a.pack_start(self.lbl_pillar_a_sub, False, False, 0)
        card_a.add(box_a)
        pillar_box.pack_start(card_a, True, True, 0)

        # Pillar B Card
        card_b = Gtk.Frame(label=" Pillar B: Valuation Evidence ")
        box_b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box_b.set_border_width(6)
        self.lbl_pillar_b_multiples = Gtk.Label()
        self.lbl_pillar_b_multiples.set_markup("<span size='medium' weight='bold' color='#555555'>P/E: — | P/B: —</span>")
        self.lbl_pillar_b_sub = Gtk.Label()
        self.lbl_pillar_b_sub.set_markup("<span size='small' color='#666666'>Decoupled from quality score</span>")
        box_b.pack_start(self.lbl_pillar_b_multiples, False, False, 0)
        box_b.pack_start(self.lbl_pillar_b_sub, False, False, 0)
        card_b.add(box_b)
        pillar_box.pack_start(card_b, True, True, 0)

        # Pillar C Card
        card_c = Gtk.Frame(label=" Pillar C: Technical Conditions ")
        box_c = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box_c.set_border_width(6)
        self.lbl_pillar_c_stance = Gtk.Label()
        self.lbl_pillar_c_stance.set_markup("<span size='medium' weight='bold' color='#555555'>Stance: —</span>")
        self.lbl_pillar_c_sub = Gtk.Label()
        self.lbl_pillar_c_sub.set_markup("<span size='small' color='#666666'>Attributed to 'Day Trading with ChatGPT'</span>")
        box_c.pack_start(self.lbl_pillar_c_stance, False, False, 0)
        box_c.pack_start(self.lbl_pillar_c_sub, False, False, 0)
        card_c.add(box_c)
        pillar_box.pack_start(card_c, True, True, 0)

        # 4-Column Card Grid of Fundamental Metrics
        self.fund_grid = Gtk.Grid()
        self.fund_grid.set_column_spacing(18)
        self.fund_grid.set_row_spacing(4)
        self.fund_grid.set_column_homogeneous(True)
        top_vbox.pack_start(self.fund_grid, False, False, 0)

        # Live metric labels
        self.lbl_val_gross_margin = Gtk.Label(label="—")
        self.lbl_val_net_margin = Gtk.Label(label="—")
        self.lbl_val_roe = Gtk.Label(label="—")
        self.lbl_val_current_ratio = Gtk.Label(label="—")
        self.lbl_val_quick_ratio = Gtk.Label(label="—")
        self.lbl_val_de_ratio = Gtk.Label(label="—")
        self.lbl_val_ocf = Gtk.Label(label="—")
        self.lbl_val_fcf = Gtk.Label(label="—")
        self.lbl_val_mktcap = Gtk.Label(label="—")
        self.lbl_val_rev_growth = Gtk.Label(label="—")
        self.lbl_val_earn_growth = Gtk.Label(label="—")
        self.lbl_val_pe = Gtk.Label(label="—")

        cards = [
            ("💰 PROFITABILITY", 0, 0, True),
            ("Gross Profit Margin:", 0, 1, False, self.lbl_val_gross_margin),
            ("Net Profit Margin:", 0, 2, False, self.lbl_val_net_margin),
            ("Return on Equity (ROE):", 0, 3, False, self.lbl_val_roe),

            ("🛡️ LIQUIDITY &amp; SOLVENCY", 1, 0, True),
            ("Current Ratio:", 1, 1, False, self.lbl_val_current_ratio),
            ("Quick Ratio:", 1, 2, False, self.lbl_val_quick_ratio),
            ("Debt-to-Equity (D/E):", 1, 3, False, self.lbl_val_de_ratio),

            ("💵 CASH FLOW STRENGTH", 2, 0, True),
            ("Operating Cash Flow:", 2, 1, False, self.lbl_val_ocf),
            ("Free Cash Flow (FCF):", 2, 2, False, self.lbl_val_fcf),
            ("Market Capitalization:", 2, 3, False, self.lbl_val_mktcap),

            ("🚀 GROWTH &amp; VALUATION", 3, 0, True),
            ("Revenue Growth (YoY):", 3, 1, False, self.lbl_val_rev_growth),
            ("Earnings Growth (YoY):", 3, 2, False, self.lbl_val_earn_growth),
            ("Trailing P/E Ratio:", 3, 3, False, self.lbl_val_pe),
        ]

        for item in cards:
            if item[3]:  # Header
                h = Gtk.Label()
                h.set_markup(f"<small><b><span color='#005a9e'>{item[0]}</span></b></small>")
                h.set_xalign(0)
                self.fund_grid.attach(h, item[1], item[2], 1, 1)
            else:
                row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
                lbl_k = Gtk.Label(label=item[0])
                lbl_k.set_xalign(0)
                val_w = item[4]
                val_w.set_xalign(1)
                row_box.pack_start(lbl_k, True, True, 0)
                row_box.pack_start(val_w, False, False, 0)
                self.fund_grid.attach(row_box, item[1], item[2], 1, 1)

        # Verdict Line
        self.lbl_fund_verdict = Gtk.Label()
        self.lbl_fund_verdict.set_markup("<b>Company Quality Verdict:</b> <span color='#777777'>No analysis loaded.</span>")
        self.lbl_fund_verdict.set_xalign(0)
        self.lbl_fund_verdict.set_line_wrap(True)
        top_vbox.pack_start(self.lbl_fund_verdict, False, False, 2)

        # =========================================================
        # 4. Notebook Tabs
        # =========================================================
        self.notebook = Gtk.Notebook()
        main_box.pack_start(self.notebook, True, True, 2)

        # Tab 0: Company Quality & Valuation (Pillars A & B)
        self.tab_quality = self._create_quality_tab()
        self.notebook.append_page(self.tab_quality, Gtk.Label(label="🏢 Company Quality & Valuation"))

        # Tab 1: Multi-Period Trends
        self.tab_trends = self._create_trends_tab()
        self.notebook.append_page(self.tab_trends, Gtk.Label(label="📈 Multi-Period Trends"))

        # Tab 2: Interactive Charts (NEW)
        self.tab_charts = self._create_charts_tab()
        self.notebook.append_page(self.tab_charts, Gtk.Label(label="📉 Interactive Charts"))

        # Tab 3: Technical Analysis Dashboard (Pillar C)
        self.tab_analysis = self._create_analysis_tab()
        self.notebook.append_page(self.tab_analysis, Gtk.Label(label="📊 Technical Analysis"))

        # Tab 4: AI Research Prompts
        self.tab_prompts = self._create_prompts_tab()
        self.notebook.append_page(self.tab_prompts, Gtk.Label(label="🤖 AI Research Prompts"))

        # Tab 5: Historical Data Table
        self.tab_data = self._create_data_tab()
        self.notebook.append_page(self.tab_data, Gtk.Label(label="📋 Historical Data Table"))

    def _select_ticker(self, symbol: str):
        self.entry_ticker.set_text(symbol)
        self.update_filename_preview()

    def update_filename_preview(self):
        ticker = self.entry_ticker.get_text().strip().upper() or "TICKER"
        days = int(self.spin_days.get_value())
        date_str = get_extraction_date_str()
        filename = f"{date_str}_{ticker}_{days}.csv"
        self.lbl_filename_preview.set_markup(f"<small>Output file: <b><tt>{filename}</tt></b></small>")

    # -------------------------------------------------------------
    # Tab 0: Company Quality & Valuation
    # -------------------------------------------------------------
    def _create_quality_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vbox.set_border_width(10)

        self.lbl_quality_caveat = Gtk.Label()
        self.lbl_quality_caveat.set_xalign(0)
        self.lbl_quality_caveat.set_line_wrap(True)
        self.lbl_quality_caveat.set_markup(
            "<span color='#555555'><i>Quality scoring uses a documented heuristic rubric. Points are not awarded for missing metrics. Below 60% data coverage, the score is withheld.</i></span>"
        )
        vbox.pack_start(self.lbl_quality_caveat, False, False, 0)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(520)
        vbox.pack_start(paned, True, True, 0)

        rubric_frame = Gtk.Frame(label=" Transparent Scoring Rubric Breakdown ")
        rubric_scroll = Gtk.ScrolledWindow()
        rubric_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.rubric_store = Gtk.ListStore(str, str, str, str, str, str)
        self.rubric_tree = Gtk.TreeView(model=self.rubric_store)
        self.rubric_tree.set_grid_lines(Gtk.TreeViewGridLines.BOTH)

        cols = [
            ("Category", 0, 95),
            ("Metric", 1, 120),
            ("Reported", 2, 75),
            ("Weight", 3, 55),
            ("Points", 4, 55),
            ("Rule Applied", 5, 120),
        ]
        for title, idx, min_w in cols:
            renderer = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(title, renderer, text=idx)
            col.set_min_width(min_w)
            col.set_resizable(True)
            self.rubric_tree.append_column(col)

        rubric_scroll.add(self.rubric_tree)
        rubric_frame.add(rubric_scroll)
        paned.pack1(rubric_frame, True, False)

        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        val_frame = Gtk.Frame(label=" Pillar B: Valuation Multiples (Decoupled from Quality) ")
        val_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        val_box.set_border_width(8)

        self.lbl_val_detail = Gtk.Label()
        self.lbl_val_detail.set_xalign(0)
        self.lbl_val_detail.set_markup("• Multiples will appear after analysis loads.")
        val_box.pack_start(self.lbl_val_detail, False, False, 0)

        lbl_val_warn = Gtk.Label()
        lbl_val_warn.set_xalign(0)
        lbl_val_warn.set_line_wrap(True)
        lbl_val_warn.set_markup(
            "<span size='small' color='#666666'><b>Core Principle:</b> Operational quality score does NOT imply shares are attractively priced. High-quality companies can trade at demanding valuations.</span>"
        )
        val_box.pack_start(lbl_val_warn, False, False, 0)
        val_frame.add(val_box)
        right_vbox.pack_start(val_frame, False, False, 0)

        narr_frame = Gtk.Frame(label=" Detailed Research Findings & Caveats ")
        narr_scroll = Gtk.ScrolledWindow()
        narr_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.quality_narrative_text = Gtk.TextView()
        self.quality_narrative_text.set_editable(False)
        self.quality_narrative_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.quality_narrative_text.set_left_margin(8)
        self.quality_narrative_text.set_right_margin(8)
        narr_scroll.add(self.quality_narrative_text)
        narr_frame.add(narr_scroll)
        right_vbox.pack_start(narr_frame, True, True, 0)

        paned.pack2(right_vbox, True, False)
        return vbox

    # -------------------------------------------------------------
    # Tab 1: Multi-Period Trends
    # -------------------------------------------------------------
    def _create_trends_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vbox.set_border_width(10)

        desc_lbl = Gtk.Label()
        desc_lbl.set_markup(
            "<b>Audited Annual Income Statement Trends:</b> "
            "Evaluates multi-year direction for Revenue, Operating Margin, and Net Profit Margin. Prevents relying on single-year snapshots."
        )
        desc_lbl.set_xalign(0)
        desc_lbl.set_line_wrap(True)
        vbox.pack_start(desc_lbl, False, False, 0)

        trend_frame = Gtk.Frame(label=" Multi-Year Annual Statement History ")
        trend_scroll = Gtk.ScrolledWindow()
        trend_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self.trends_store = Gtk.ListStore(str, str, str, str, str, str, str, str, str)
        self.trends_tree = Gtk.TreeView(model=self.trends_store)
        self.trends_tree.set_grid_lines(Gtk.TreeViewGridLines.BOTH)

        trend_cols = [
            ("Fiscal Period", 0, 110),
            ("Filing Date", 1, 90),
            ("Revenue", 2, 85),
            ("Gross Profit", 3, 85),
            ("Gross Margin", 4, 85),
            ("Operating Inc", 5, 90),
            ("Operating Margin", 6, 95),
            ("Net Income", 7, 85),
            ("Net Margin", 8, 85),
        ]
        for title, idx, min_w in trend_cols:
            renderer = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(title, renderer, text=idx)
            col.set_min_width(min_w)
            col.set_resizable(True)
            self.trends_tree.append_column(col)

        trend_scroll.add(self.trends_tree)
        trend_frame.add(trend_scroll)
        vbox.pack_start(trend_frame, True, True, 0)

        self.lbl_trend_summary = Gtk.Label()
        self.lbl_trend_summary.set_xalign(0)
        self.lbl_trend_summary.set_line_wrap(True)
        self.lbl_trend_summary.set_markup("<span color='#555555'>Trend observations will populate after data download.</span>")
        vbox.pack_start(self.lbl_trend_summary, False, False, 0)

        return vbox

    # -------------------------------------------------------------
    # Tab 2: Interactive Charts (Embedded Matplotlib in GTK 3)
    # -------------------------------------------------------------
    def _create_charts_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        vbox.set_border_width(8)

        # Control Row 1: Interval, View Type, Overlays
        ctrl_box1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        vbox.pack_start(ctrl_box1, False, False, 2)

        lbl_int = Gtk.Label(label="Interval:")
        ctrl_box1.pack_start(lbl_int, False, False, 0)

        self.combo_chart_interval = Gtk.ComboBoxText()
        self.combo_chart_interval.append("daily", "Daily Bars")
        self.combo_chart_interval.append("weekly", "Weekly Bars")
        self.combo_chart_interval.set_active_id("daily")
        self.combo_chart_interval.connect("changed", self._on_chart_interval_changed)
        ctrl_box1.pack_start(self.combo_chart_interval, False, False, 0)

        lbl_type = Gtk.Label(label="Style:")
        ctrl_box1.pack_start(lbl_type, False, False, 0)

        self.combo_chart_type = Gtk.ComboBoxText()
        self.combo_chart_type.append("candlestick", "Candlestick")
        self.combo_chart_type.append("line", "Line (Close)")
        self.combo_chart_type.set_active_id("candlestick")
        self.combo_chart_type.connect("changed", self._on_chart_type_changed)
        ctrl_box1.pack_start(self.combo_chart_type, False, False, 0)

        ctrl_box1.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        lbl_ov = Gtk.Label(label="Overlays:")
        lbl_ov.get_style_context().add_class("dim-label")
        ctrl_box1.pack_start(lbl_ov, False, False, 0)

        # Default visible: SMA-20, SMA-50
        self.chk_sma20 = Gtk.CheckButton(label="SMA 20")
        self.chk_sma20.set_active(True)
        self.chk_sma20.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_sma20, False, False, 0)

        self.chk_sma50 = Gtk.CheckButton(label="SMA 50")
        self.chk_sma50.set_active(True)
        self.chk_sma50.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_sma50, False, False, 0)

        self.chk_sma200 = Gtk.CheckButton(label="SMA 200")
        self.chk_sma200.set_active(False)
        self.chk_sma200.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_sma200, False, False, 0)

        self.chk_ema12 = Gtk.CheckButton(label="EMA 12")
        self.chk_ema12.set_active(False)
        self.chk_ema12.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_ema12, False, False, 0)

        self.chk_ema26 = Gtk.CheckButton(label="EMA 26")
        self.chk_ema26.set_active(False)
        self.chk_ema26.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_ema26, False, False, 0)

        self.chk_bollinger = Gtk.CheckButton(label="Bollinger Bands")
        self.chk_bollinger.set_active(False)
        self.chk_bollinger.connect("toggled", lambda w: self._render_chart())
        ctrl_box1.pack_start(self.chk_bollinger, False, False, 0)

        # Control Row 2: Panels, Range Buttons, Export PNG
        ctrl_box2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        vbox.pack_start(ctrl_box2, False, False, 2)

        lbl_pan = Gtk.Label(label="Panels:")
        lbl_pan.get_style_context().add_class("dim-label")
        ctrl_box2.pack_start(lbl_pan, False, False, 0)

        # Volume default visible; Oscillators optional
        self.chk_panel_vol = Gtk.CheckButton(label="Volume")
        self.chk_panel_vol.set_active(True)
        self.chk_panel_vol.connect("toggled", lambda w: self._render_chart())
        ctrl_box2.pack_start(self.chk_panel_vol, False, False, 0)

        self.chk_panel_rsi = Gtk.CheckButton(label="RSI (14)")
        self.chk_panel_rsi.set_active(False)
        self.chk_panel_rsi.connect("toggled", lambda w: self._render_chart())
        ctrl_box2.pack_start(self.chk_panel_rsi, False, False, 0)

        self.chk_panel_macd = Gtk.CheckButton(label="MACD (12,26,9)")
        self.chk_panel_macd.set_active(False)
        self.chk_panel_macd.connect("toggled", lambda w: self._render_chart())
        ctrl_box2.pack_start(self.chk_panel_macd, False, False, 0)

        self.chk_panel_stoch = Gtk.CheckButton(label="Stochastic (14,3)")
        self.chk_panel_stoch.set_active(False)
        self.chk_panel_stoch.connect("toggled", lambda w: self._render_chart())
        ctrl_box2.pack_start(self.chk_panel_stoch, False, False, 0)

        ctrl_box2.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        lbl_rng = Gtk.Label(label="Range:")
        lbl_rng.get_style_context().add_class("dim-label")
        ctrl_box2.pack_start(lbl_rng, False, False, 0)

        range_buttons = [("1M", "1m"), ("3M", "3m"), ("6M", "6m"), ("1Y", "1y"), ("All", "all")]
        for lbl_r, r_val in range_buttons:
            btn_r = Gtk.Button(label=lbl_r)
            btn_r.set_relief(Gtk.ReliefStyle.NONE)
            btn_r.connect("clicked", lambda b, rv=r_val: self._on_range_clicked(rv))
            ctrl_box2.pack_start(btn_r, False, False, 0)

        ctrl_box2.pack_start(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL), False, False, 4)

        btn_export_png = Gtk.Button(label="💾 Export PNG")
        btn_export_png.get_style_context().add_class("suggested-action")
        btn_export_png.connect("clicked", self.on_export_chart_png)
        ctrl_box2.pack_start(btn_export_png, False, False, 0)

        # Crosshair & Hover Readout Label
        self.lbl_crosshair_readout = Gtk.Label()
        self.lbl_crosshair_readout.set_xalign(0)
        self.lbl_crosshair_readout.set_markup("<span color='#666666'>Hover cursor over chart for crosshair, session OHLC, volume, and indicator readings.</span>")
        vbox.pack_start(self.lbl_crosshair_readout, False, False, 2)

        # Embedded Matplotlib Figure and Canvas
        self.chart_fig = Figure(figsize=(10, 5.8), dpi=100)
        self.chart_canvas = FigureCanvasGTK3Agg(self.chart_fig)
        self.chart_canvas.set_size_request(600, 380)

        self.chart_toolbar = NavigationToolbar2GTK3(self.chart_canvas)
        vbox.pack_start(self.chart_toolbar, False, False, 0)

        chart_frame = Gtk.Frame()
        chart_frame.add(self.chart_canvas)
        vbox.pack_start(chart_frame, True, True, 0)

        # Crosshair motion connect
        self.chart_canvas.mpl_connect("motion_notify_event", self._on_chart_mouse_move)

        # Correlation disclaimer footnote
        lbl_disclaimer = Gtk.Label()
        lbl_disclaimer.set_xalign(0)
        lbl_disclaimer.set_line_wrap(True)
        lbl_disclaimer.set_markup(
            "<span size='small' color='#777777'><b>Correlation Notice:</b> Technical indicators reflect historical price and volume transformations. Multiple momentum oscillators sharing common inputs (e.g. RSI, MACD, Stochastic) do not provide independent evidence.</span>"
        )
        vbox.pack_start(lbl_disclaimer, False, False, 2)

        # Initial render of clean empty state
        self._render_empty_chart()
        return vbox

    def _render_empty_chart(self, message: str = "No analysis loaded.\nEnter a ticker and click 'Download & Analyze' to view interactive charts."):
        self.chart_fig.clear()
        ax = self.chart_fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#777777", transform=ax.transAxes)
        ax.set_axis_off()
        self.chart_canvas.draw_idle()

    def _on_chart_interval_changed(self, widget):
        self.current_chart_interval = self.combo_chart_interval.get_active_id() or "daily"
        self._render_chart()

    def _on_chart_type_changed(self, widget):
        self.current_chart_type = self.combo_chart_type.get_active_id() or "candlestick"
        self._render_chart()

    def _on_range_clicked(self, range_val: str):
        self.current_chart_range = range_val
        self._apply_chart_range()

    def _render_chart(self):
        if not self.full_history_records:
            self._render_empty_chart()
            return

        ticker = self.current_ticker or "TICKER"
        currency = self.current_fundamentals.get("currency", "USD") if self.current_fundamentals else "USD"

        # Determine interval data
        if self.current_chart_interval == "weekly":
            if not self.weekly_history_records:
                self.weekly_history_records = aggregate_to_weekly(self.full_history_records)
                self.weekly_analysis = analyze_stock_data(self.weekly_history_records, ticker, calculation_history=self.weekly_history_records)
            records = self.weekly_history_records
            analysis = self.weekly_analysis or {}
            interval_label = "Weekly"
        else:
            records = self.full_history_records
            analysis = self.current_analysis or {}
            interval_label = "Daily"

        if not records:
            self._render_empty_chart("No bars available for this interval.")
            return

        # Ensure series is available
        series = analysis.get("series")
        if not series or len(series.get("closes", [])) == 0:
            analysis = analyze_stock_data(records, ticker, calculation_history=records)
            series = analysis.get("series", {})

        self.chart_active_series = series
        dates = series.get("dates", [])
        closes = series.get("closes", [])
        opens = series.get("opens", [])
        highs = series.get("highs", [])
        lows = series.get("lows", [])
        volumes = series.get("volumes", [])
        N = len(dates)

        if N == 0:
            self._render_empty_chart("No price records to plot.")
            return

        # Determine active panels
        active_panels = []
        if self.chk_panel_vol.get_active():
            active_panels.append("volume")
        if self.chk_panel_rsi.get_active():
            active_panels.append("rsi")
        if self.chk_panel_macd.get_active():
            active_panels.append("macd")
        if self.chk_panel_stoch.get_active():
            active_panels.append("stoch")

        nrows = 1 + len(active_panels)
        height_ratios = [3.0] + [0.9] * len(active_panels)

        self.chart_fig.clear()
        sub_axes = self.chart_fig.subplots(nrows, 1, sharex=True, gridspec_kw={"height_ratios": height_ratios})
        if nrows == 1:
            axes = [sub_axes]
        else:
            axes = list(sub_axes)
        self.chart_axes = axes
        ax_price = axes[0]

        panel_axes = {}
        for idx, p_name in enumerate(active_panels):
            panel_axes[p_name] = axes[idx + 1]

        x = np.arange(N)

        # 1. Price Chart
        if self.current_chart_type == "candlestick":
            for i in range(N):
                c = closes[i]
                o = opens[i]
                h = highs[i]
                l = lows[i]
                col = "#26a69a" if c >= o else "#ef5350"
                ax_price.plot([i, i], [l, h], color=col, linewidth=0.9)
                body_b = min(o, c)
                body_h = max(abs(c - o), 0.01)
                ax_price.bar(i, body_h, bottom=body_b, color=col, width=0.64, align="center", edgecolor=col)
        else:
            ax_price.plot(x, closes, color="#1976d2", linewidth=1.5, label="Close")
            ax_price.fill_between(x, closes, alpha=0.08, color="#1976d2")

        # Overlays
        if self.chk_sma20.get_active() and "sma_20" in series:
            s20 = [v for v in series["sma_20"]]
            ax_price.plot(x, s20, color="#ff9800", linewidth=1.1, label="SMA 20")
        if self.chk_sma50.get_active() and "sma_50" in series:
            s50 = [v for v in series["sma_50"]]
            ax_price.plot(x, s50, color="#2196f3", linewidth=1.2, label="SMA 50")
        if self.chk_sma200.get_active() and "sma_200" in series:
            s200 = [v for v in series["sma_200"]]
            ax_price.plot(x, s200, color="#e91e63", linewidth=1.3, label="SMA 200")
        if self.chk_ema12.get_active() and "ema_12" in series:
            e12 = [v for v in series["ema_12"]]
            ax_price.plot(x, e12, color="#00bcd4", linewidth=1.0, linestyle="--", label="EMA 12")
        if self.chk_ema26.get_active() and "ema_26" in series:
            e26 = [v for v in series["ema_26"]]
            ax_price.plot(x, e26, color="#9c27b0", linewidth=1.0, linestyle="--", label="EMA 26")
        if self.chk_bollinger.get_active() and "bb_upper" in series:
            b_u = series["bb_upper"]
            b_l = series["bb_lower"]
            b_m = series["bb_mid"]
            ax_price.plot(x, b_m, color="#78909c", linewidth=0.8, linestyle=":")
            ax_price.plot(x, b_u, color="#607d8b", linewidth=0.9, label="Bollinger")
            ax_price.plot(x, b_l, color="#607d8b", linewidth=0.9)
            valid_idx = [i for i in range(N) if b_u[i] is not None and b_l[i] is not None]
            if valid_idx:
                v_x = x[valid_idx]
                v_u = [b_u[i] for i in valid_idx]
                v_l = [b_l[i] for i in valid_idx]
                ax_price.fill_between(v_x, v_l, v_u, color="#607d8b", alpha=0.07)

        ax_price.grid(True, linestyle=":", alpha=0.45)
        ax_price.legend(loc="upper left", fontsize=8, framealpha=0.6)

        # Title & provenance
        latest_date = dates[-1]
        latest_c = closes[-1]
        is_today = (latest_date == datetime.date.today().strftime("%Y-%m-%d"))
        session_flag = " [Intraday / Incomplete]" if is_today else ""
        ax_price.set_title(
            f"{ticker} ({currency}) • {interval_label} | Latest: {latest_date} ({latest_c:.2f}){session_flag} | Policy: Unadjusted / Split-Adjusted",
            fontsize=9.5,
            fontweight="bold",
            loc="left",
            pad=6,
        )

        # 2. Volume Panel
        if "volume" in panel_axes:
            ax_vol = panel_axes["volume"]
            vol_colors = ["#26a69a" if closes[i] >= opens[i] else "#ef5350" for i in range(N)]
            ax_vol.bar(x, volumes, color=vol_colors, width=0.65, alpha=0.75)
            ax_vol.set_ylabel("Vol", fontsize=8)
            ax_vol.grid(True, linestyle=":", alpha=0.4)

        # 3. RSI Panel
        if "rsi" in panel_axes:
            ax_rsi = panel_axes["rsi"]
            rsi_vals = series.get("rsi_14", [])
            ax_rsi.plot(x, rsi_vals, color="#ab47bc", linewidth=1.2, label="RSI 14")
            ax_rsi.axhline(70, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.8)
            ax_rsi.axhline(30, color="#26a69a", linestyle="--", linewidth=0.8, alpha=0.8)
            ax_rsi.fill_between(x, 30, 70, color="#ab47bc", alpha=0.05)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_yticks([30, 70])
            ax_rsi.set_ylabel("RSI", fontsize=8)
            ax_rsi.grid(True, linestyle=":", alpha=0.4)

        # 4. MACD Panel
        if "macd" in panel_axes:
            ax_macd = panel_axes["macd"]
            m_line = series.get("macd_line", [])
            m_sig = series.get("macd_sig", [])
            m_hist = series.get("macd_hist", [])
            ax_macd.plot(x, m_line, color="#2196f3", linewidth=1.1, label="MACD")
            ax_macd.plot(x, m_sig, color="#ff9800", linewidth=1.1, label="Signal")
            hist_cols = ["#26a69a" if (h or 0) >= 0 else "#ef5350" for h in m_hist]
            ax_macd.bar(x, [h if h is not None else 0 for h in m_hist], color=hist_cols, width=0.55, alpha=0.75)
            ax_macd.axhline(0, color="#888888", linestyle=":", linewidth=0.7)
            ax_macd.set_ylabel("MACD", fontsize=8)
            ax_macd.grid(True, linestyle=":", alpha=0.4)

        # 5. Stochastic Panel
        if "stoch" in panel_axes:
            ax_stoch = panel_axes["stoch"]
            s_k = series.get("stoch_k", [])
            s_d = series.get("stoch_d", [])
            ax_stoch.plot(x, s_k, color="#0288d1", linewidth=1.1, label="%K")
            ax_stoch.plot(x, s_d, color="#f57c00", linewidth=1.1, label="%D")
            ax_stoch.axhline(80, color="#ef5350", linestyle="--", linewidth=0.8, alpha=0.8)
            ax_stoch.axhline(20, color="#26a69a", linestyle="--", linewidth=0.8, alpha=0.8)
            ax_stoch.fill_between(x, 20, 80, color="#f57c00", alpha=0.05)
            ax_stoch.set_ylim(0, 100)
            ax_stoch.set_yticks([20, 80])
            ax_stoch.set_ylabel("Stoch", fontsize=8)
            ax_stoch.grid(True, linestyle=":", alpha=0.4)

        # X-Axis tick formatting on the bottom-most axis
        for ax_elem in axes[:-1]:
            ax_elem.tick_params(labelbottom=False)

        bottom_ax = axes[-1]
        step = max(1, N // 8)
        tick_pos = list(range(0, N, step))
        if tick_pos[-1] != N - 1 and (N - 1 - tick_pos[-1]) > step // 2:
            tick_pos.append(N - 1)
        bottom_ax.set_xticks(tick_pos)
        bottom_ax.set_xticklabels([dates[i] for i in tick_pos], rotation=22, ha="right", fontsize=8)

        self.chart_fig.subplots_adjust(hspace=0.06, left=0.07, right=0.96, top=0.93, bottom=0.10)
        self._apply_chart_range(draw=False)
        self.chart_canvas.draw_idle()

    def _apply_chart_range(self, draw: bool = True):
        if not hasattr(self, "chart_axes") or len(self.chart_axes) == 0:
            return
        series = self.chart_active_series
        dates = series.get("dates", [])
        N = len(dates)
        if N == 0:
            return

        interval = self.current_chart_interval
        # Mapping ranges to bars
        if self.current_chart_range == "1m":
            bars = 21 if interval == "daily" else 4
        elif self.current_chart_range == "3m":
            bars = 63 if interval == "daily" else 13
        elif self.current_chart_range == "6m":
            bars = 126 if interval == "daily" else 26
        elif self.current_chart_range == "1y":
            bars = 252 if interval == "daily" else 52
        else:
            bars = N

        ax_price = self.chart_axes[0]
        if bars < N:
            x_min = N - bars - 0.5
            x_max = N - 0.5
            ax_price.set_xlim(x_min, x_max)
            # Rescale price y-limits to visible high/low
            vis_h = [series["highs"][i] for i in range(N - bars, N)]
            vis_l = [series["lows"][i] for i in range(N - bars, N)]
            if vis_h and vis_l:
                pad = max((max(vis_h) - min(vis_l)) * 0.05, 0.5)
                ax_price.set_ylim(min(vis_l) - pad, max(vis_h) + pad)
        else:
            ax_price.set_xlim(-0.5, N - 0.5)
            h = series.get("highs", [])
            l = series.get("lows", [])
            if h and l:
                pad = max((max(h) - min(l)) * 0.05, 0.5)
                ax_price.set_ylim(min(l) - pad, max(h) + pad)

        if draw:
            self.chart_canvas.draw_idle()

    def _on_chart_mouse_move(self, event):
        if event.inaxes is None or event.xdata is None or not self.chart_active_series:
            return
        series = self.chart_active_series
        dates = series.get("dates", [])
        N = len(dates)
        if N == 0:
            return

        idx = int(round(event.xdata))
        if 0 <= idx < N:
            d = dates[idx]
            o = series["opens"][idx]
            h = series["highs"][idx]
            l = series["lows"][idx]
            c = series["closes"][idx]
            v = series["volumes"][idx]

            readout_parts = [
                f"<b>Date:</b> {d}",
                f"<b>O:</b> {o:.2f}",
                f"<b>H:</b> {h:.2f}",
                f"<b>L:</b> {l:.2f}",
                f"<b>C:</b> {c:.2f}",
                f"<b>Vol:</b> {v:,.0f}",
            ]
            if self.chk_sma20.get_active() and series.get("sma_20") and series["sma_20"][idx] is not None:
                readout_parts.append(f"<b>SMA20:</b> {series['sma_20'][idx]:.2f}")
            if self.chk_sma50.get_active() and series.get("sma_50") and series["sma_50"][idx] is not None:
                readout_parts.append(f"<b>SMA50:</b> {series['sma_50'][idx]:.2f}")
            if self.chk_panel_rsi.get_active() and series.get("rsi_14") and series["rsi_14"][idx] is not None:
                readout_parts.append(f"<b>RSI:</b> {series['rsi_14'][idx]:.1f}")
            if self.chk_panel_macd.get_active() and series.get("macd_line") and series["macd_line"][idx] is not None:
                readout_parts.append(f"<b>MACD:</b> {series['macd_line'][idx]:.2f}")
            if self.chk_panel_stoch.get_active() and series.get("stoch_k") and series["stoch_k"][idx] is not None:
                readout_parts.append(f"<b>%K:</b> {series['stoch_k'][idx]:.1f}")

            self.lbl_crosshair_readout.set_markup("  |  ".join(readout_parts))

    def on_export_chart_png(self, widget):
        if not self.full_history_records:
            self.show_error("No chart data loaded to export.")
            return

        ticker = self.current_ticker or "TICKER"
        interval = self.current_chart_interval
        date_str = get_extraction_date_str()
        default_name = f"{date_str}_{ticker}_{interval}_chart.png"

        dialog = Gtk.FileChooserDialog(
            title="Export Chart as PNG",
            parent=self,
            action=Gtk.FileChooserAction.SAVE,
        )
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT)
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_name(default_name)
        dialog.set_current_folder(self.file_chooser.get_filename() or os.getcwd())

        resp = dialog.run()
        if resp == Gtk.ResponseType.ACCEPT:
            save_path = dialog.get_filename()
            if not save_path.lower().endswith(".png"):
                save_path += ".png"
            try:
                self.chart_fig.savefig(save_path, dpi=180, bbox_inches="tight")
                self.status_label.set_markup(f"<span color='#0e8a16'>✔ Chart PNG exported → <b>{os.path.basename(save_path)}</b></span>")
            except Exception as e:
                self.show_error(f"Failed to export chart PNG: {e}")
        dialog.destroy()

    # -------------------------------------------------------------
    # Tab 3: Technical Analysis Dashboard (Pillar C)
    # -------------------------------------------------------------
    def _create_analysis_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vbox.set_border_width(8)

        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        vbox.pack_start(top_row, False, False, 0)

        # Technical Stance Card
        stance_frame = Gtk.Frame(label=" Technical Stance ")
        stance_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        stance_box.set_border_width(8)
        self.lbl_stance_badge = Gtk.Label()
        self.lbl_stance_badge.set_markup("<span size='xx-large' weight='bold'>—</span>")
        self.lbl_stance_detail = Gtk.Label(label="No analysis loaded.")
        self.lbl_stance_detail.set_line_wrap(True)
        stance_box.pack_start(self.lbl_stance_badge, True, True, 0)
        stance_box.pack_start(self.lbl_stance_detail, False, False, 0)
        stance_frame.add(stance_box)
        stance_frame.set_size_request(240, -1)
        top_row.pack_start(stance_frame, False, False, 0)

        # Technical Snapshot Metrics
        tech_frame = Gtk.Frame(label=" Technical Indicator Snapshot (from 'Day Trading with ChatGPT') ")
        grid = Gtk.Grid()
        grid.set_column_spacing(16)
        grid.set_row_spacing(5)
        grid.set_border_width(8)

        self.lbl_m_rsi = Gtk.Label(label="—")
        self.lbl_m_macd = Gtk.Label(label="—")
        self.lbl_m_bb = Gtk.Label(label="—")
        self.lbl_m_ma = Gtk.Label(label="—")
        self.lbl_m_stoch = Gtk.Label(label="—")
        self.lbl_m_fib = Gtk.Label(label="—")

        items = [
            ("RSI (14-day):", self.lbl_m_rsi, 0, 0),
            ("MACD (12, 26, 9):", self.lbl_m_macd, 1, 0),
            ("Bollinger Bands:", self.lbl_m_bb, 0, 1),
            ("Moving Averages:", self.lbl_m_ma, 1, 1),
            ("Stochastic (%K/%D):", self.lbl_m_stoch, 0, 2),
            ("Fibonacci S/R:", self.lbl_m_fib, 1, 2),
        ]
        for title, val_lbl, col, row in items:
            t = Gtk.Label()
            t.set_markup(f"<b>{title}</b>")
            t.set_xalign(0)
            val_lbl.set_xalign(0)
            grid.attach(t, col * 2, row, 1, 1)
            grid.attach(val_lbl, col * 2 + 1, row, 1, 1)

        tech_frame.add(grid)
        top_row.pack_start(tech_frame, True, True, 0)

        # Paned: Signals & Report
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(380)
        vbox.pack_start(paned, True, True, 0)

        # Signals List
        sig_frame = Gtk.Frame(label=" Actionable Signals & Crossover Triggers ")
        sig_scroll = Gtk.ScrolledWindow()
        sig_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.signals_store = Gtk.ListStore(str, str, str)
        self.signals_tree = Gtk.TreeView(model=self.signals_store)
        self.signals_tree.set_grid_lines(Gtk.TreeViewGridLines.BOTH)

        col_ind = Gtk.TreeViewColumn("Indicator", Gtk.CellRendererText(), text=0)
        col_ind.set_min_width(90)
        self.signals_tree.append_column(col_ind)

        col_bias = Gtk.TreeViewColumn("Bias", Gtk.CellRendererText(), text=1)
        col_bias.set_min_width(85)
        self.signals_tree.append_column(col_bias)

        col_desc = Gtk.TreeViewColumn("Signal Details", Gtk.CellRendererText(), text=2)
        col_desc.set_resizable(True)
        self.signals_tree.append_column(col_desc)

        sig_scroll.add(self.signals_tree)
        sig_frame.add(sig_scroll)
        paned.pack1(sig_frame, True, False)

        # Synthesis Report
        rep_frame = Gtk.Frame(label=" Full Technical Synthesis (Attributed to Saskia Adler) ")
        rep_scroll = Gtk.ScrolledWindow()
        rep_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.report_text = Gtk.TextView()
        self.report_text.set_editable(False)
        self.report_text.set_wrap_mode(Gtk.WrapMode.WORD)
        self.report_text.set_left_margin(8)
        self.report_text.set_right_margin(8)
        rep_scroll.add(self.report_text)
        rep_frame.add(rep_scroll)
        paned.pack2(rep_frame, True, False)

        return vbox

    # -------------------------------------------------------------
    # Tab 4: AI Research Prompts
    # -------------------------------------------------------------
    def _create_prompts_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        vbox.set_border_width(10)

        info_lbl = Gtk.Label()
        info_lbl.set_markup(
            "<span>Select a research prompt template below. All prompts embed verified snapshot data with strict anti-hallucination and investment horizon boundaries:</span>"
        )
        info_lbl.set_xalign(0)
        info_lbl.set_line_wrap(True)
        vbox.pack_start(info_lbl, False, False, 0)

        sel_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        vbox.pack_start(sel_box, False, False, 0)

        lbl_cat = Gtk.Label(label="Template:")
        sel_box.pack_start(lbl_cat, False, False, 0)

        self.combo_prompts = Gtk.ComboBoxText()
        templates = [
            ("quality", "Pillar A: Company Financial Quality & Moats"),
            ("valuation", "Pillar B: Valuation Multiples & Growth Expectations"),
            ("technicals", "Pillar C: Technical Setup & Swing Points"),
            ("synthesis", "Multi-Pillar Executive Investment Synthesis"),
            ("Moving Averages (Chapter 3)", "Book Ch 3: Moving Averages & Crossovers"),
            ("RSI & Momentum (Chapter 4)", "Book Ch 4: RSI Momentum & Divergences"),
            ("Bollinger Bands (Chapter 5)", "Book Ch 5: Bollinger Bands & Volatility"),
            ("Fibonacci Retracement (Chapter 6)", "Book Ch 6: Fibonacci Retracements"),
            ("MACD & Trend Momentum (Chapter 7)", "Book Ch 7: MACD Trend & Momentum"),
            ("Stochastic Oscillator (Chapter 8)", "Book Ch 8: Stochastic Oscillator"),
        ]
        for key, name in templates:
            self.combo_prompts.append(key, name)
        self.combo_prompts.set_active_id("synthesis")
        self.combo_prompts.connect("changed", self.on_prompt_selection_changed)
        sel_box.pack_start(self.combo_prompts, True, True, 0)

        btn_copy = Gtk.Button(label="📋 Copy Prompt to Clipboard")
        btn_copy.connect("clicked", self.on_copy_prompt_clicked)
        sel_box.pack_start(btn_copy, False, False, 0)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.prompt_text_view = Gtk.TextView()
        self.prompt_text_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.prompt_text_view.set_left_margin(10)
        self.prompt_text_view.set_right_margin(10)
        self.prompt_text_view.get_buffer().set_text("No analysis loaded. Download and analyze data to generate verified research prompts.")
        scrolled.add(self.prompt_text_view)
        vbox.pack_start(scrolled, True, True, 0)

        return vbox

    # -------------------------------------------------------------
    # Tab 5: Historical Data Table
    # -------------------------------------------------------------
    def _create_data_tab(self) -> Gtk.Widget:
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_border_width(6)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        vbox.pack_start(scrolled, True, True, 0)

        self.treeview = Gtk.TreeView()
        self.treeview.set_grid_lines(Gtk.TreeViewGridLines.BOTH)
        scrolled.add(self.treeview)
        self.setup_table_columns(include_indicators=True)

        return vbox

    def setup_table_columns(self, include_indicators: bool = True):
        for col in list(self.treeview.get_columns()):
            self.treeview.remove_column(col)

        if include_indicators:
            col_names = [
                "Date", "Close", "Open", "High", "Low", "Volume",
                "SMA-20", "RSI-14", "BB-Upper", "BB-Lower", "MACD", "Stoch-%K"
            ]
        else:
            col_names = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        types = [str] * len(col_names)
        self.liststore = Gtk.ListStore(*types)
        self.treeview.set_model(self.liststore)

        for i, name in enumerate(col_names):
            renderer = Gtk.CellRendererText()
            if i > 0:
                renderer.set_alignment(1.0, 0.5)
            col = Gtk.TreeViewColumn(name, renderer, text=i)
            col.set_resizable(True)
            col.set_min_width(75)
            self.treeview.append_column(col)

    # -------------------------------------------------------------
    # Execution & Event Handlers (Worker Threads & State Updates)
    # -------------------------------------------------------------
    def on_download_clicked(self, widget):
        if self.download_in_progress:
            return

        ticker = self.entry_ticker.get_text().strip().upper()
        if not ticker:
            self.show_error("Please enter a stock ticker symbol (e.g. NVDA).")
            return

        days = int(self.spin_days.get_value())
        if days <= 0:
            self.show_error("Please enter a positive number of trading days.")
            return

        out_dir = self.file_chooser.get_filename() or os.getcwd()
        sort_order = self.combo_sort.get_active_id() or "desc"
        include_indicators = self.chk_include_indicators.get_active()

        self.active_request_token += 1
        current_token = self.active_request_token
        self.current_ticker = ticker

        self.download_in_progress = True
        self.btn_download.set_sensitive(False)
        self.btn_open_file.set_sensitive(False)
        self.btn_open_folder.set_sensitive(False)
        self.spinner.start()
        self.status_label.set_markup(f"<i>Fetching market &amp; financial data for {ticker} ({days} days)...</i>")

        worker = threading.Thread(
            target=self._worker_download_and_analyze,
            args=(current_token, ticker, days, out_dir, sort_order, include_indicators),
            daemon=True,
        )
        worker.start()

    def _worker_download_and_analyze(
        self,
        token: int,
        ticker: str,
        days: int,
        out_dir: str,
        sort_order: str,
        include_indicators: bool,
    ):
        price_error = None
        fund_error = None
        csv_path = None
        rows = []
        full_records = []
        analysis = None
        fund = None

        # 1. Fetch prices & compute chronological technicals
        try:
            csv_path, rows, full_records = download_stock_prices(
                ticker=ticker,
                trading_days=days,
                output_dir=out_dir,
                sort_order=sort_order,
                close_only=False,
                include_no_ext_link=True,
                include_indicators=include_indicators,
                return_full_history=True,
            )
            analysis = analyze_stock_data(rows, ticker, calculation_history=full_records)
        except Exception as e:
            price_error = str(e)

        # 2. Fetch company fundamentals
        try:
            fund = fetch_company_fundamentals(ticker)
            if "error" in fund and not price_error:
                fund_error = fund.get("error")
        except Exception as e:
            fund_error = str(e)

        GLib.idle_add(
            self._on_download_complete,
            token,
            ticker,
            csv_path,
            rows,
            full_records,
            analysis,
            fund,
            price_error,
            fund_error,
            include_indicators,
        )

    def _on_download_complete(
        self,
        token: int,
        ticker: str,
        csv_path: Optional[str],
        rows: List[Dict],
        full_records: List[Dict],
        analysis: Optional[Dict[str, Any]],
        fund: Optional[Dict[str, Any]],
        price_error: Optional[str],
        fund_error: Optional[str],
        include_indicators: bool,
    ):
        if token != self.active_request_token:
            return False

        self.download_in_progress = False
        self.spinner.stop()
        self.btn_download.set_sensitive(True)

        if price_error and (fund_error or fund is None):
            err_msg = f"Price download failed: {price_error}. Fundamentals failed: {fund_error}"
            self._set_state_failed(ticker, err_msg)
            return False

        if price_error:
            self.status_label.set_markup(
                f"<span color='#d73a49'>❌ Price download failed for <b>{ticker}</b>: {GLib.markup_escape_text(price_error)}</span>"
            )
            if fund and "error" not in fund:
                self._update_fundamentals_ui(fund)
            self._render_empty_chart(f"Chart unavailable: {price_error}")
            return False

        # Price download succeeded!
        self.last_saved_file = csv_path
        self.current_rows = rows
        # Chronological sort for chart engine
        self.full_history_records = sorted(full_records, key=lambda r: r.get("Date", ""))
        self.weekly_history_records = []
        self.weekly_analysis = None
        self.current_analysis = analysis
        self.btn_open_file.set_sensitive(True)
        self.btn_open_folder.set_sensitive(True)

        basename = os.path.basename(csv_path) if csv_path else "export.csv"
        if fund_error or (fund and "error" in fund):
            actual_err = fund_error or fund.get("error", "Unknown error")
            self.status_label.set_markup(
                f"<span color='#b08800'>⚠️ Downloaded prices ({len(rows)} days) for <b>{ticker}</b>. Fundamentals unavailable: {GLib.markup_escape_text(actual_err)}</span>"
            )
            self._set_fundamentals_unavailable(ticker, actual_err)
        else:
            self.status_label.set_markup(
                f"<span color='#0e8a16'>✔ Downloaded {len(rows)} days for <b>{ticker}</b> → {basename}</span>"
            )
            if fund:
                self._update_fundamentals_ui(fund)

        if analysis:
            self._update_technicals_ui(analysis)

        # Render Interactive Charts Tab
        self._render_chart()

        self._populate_table(rows, include_indicators)
        self.on_prompt_selection_changed(None)
        return False

    def _set_state_failed(self, ticker: str, err_msg: str):
        self.status_label.set_markup(f"<span color='#d73a49'>❌ {GLib.markup_escape_text(err_msg)}</span>")
        self.lbl_fund_company.set_markup(f"<b>Failed to load data for {GLib.markup_escape_text(ticker)}</b>")
        self.lbl_fund_badge.set_markup("<span color='#d73a49'>● ERROR</span>")
        self.lbl_fund_verdict.set_markup(f"<span color='#d73a49'>Request error: {GLib.markup_escape_text(err_msg)}</span>")
        self.lbl_pillar_a_score.set_markup("<span color='#d73a49'>Score: N/A</span>")
        self.lbl_pillar_b_multiples.set_markup("<span color='#d73a49'>Multiples: N/A</span>")
        self.lbl_pillar_c_stance.set_markup("<span color='#d73a49'>Stance: N/A</span>")
        self._render_empty_chart(f"Chart unavailable: {err_msg}")

    def _set_fundamentals_unavailable(self, ticker: str, error_msg: str):
        self.current_fundamentals = None
        self.lbl_fund_company.set_markup(f"<b>{GLib.markup_escape_text(ticker)}</b> | Fundamental Data Unavailable")
        self.lbl_fund_badge.set_markup("<span color='#888888'>⚪ No Fundamentals</span>")
        self.lbl_fund_verdict.set_markup(
            f"<span color='#b08800'>Provider could not supply financial indicators for {GLib.markup_escape_text(ticker)}: {GLib.markup_escape_text(error_msg)}</span>"
        )
        self.lbl_pillar_a_score.set_markup("<span color='#888888'>Score: WITHHELD</span>")
        self.lbl_pillar_a_sub.set_markup("<span size='small' color='#888888'>Coverage: 0.0%</span>")
        self.lbl_pillar_b_multiples.set_markup("<span color='#888888'>Multiples: N/A</span>")

        for lbl in [
            self.lbl_val_gross_margin, self.lbl_val_net_margin, self.lbl_val_roe,
            self.lbl_val_current_ratio, self.lbl_val_quick_ratio, self.lbl_val_de_ratio,
            self.lbl_val_ocf, self.lbl_val_fcf, self.lbl_val_mktcap,
            self.lbl_val_rev_growth, self.lbl_val_earn_growth, self.lbl_val_pe,
        ]:
            lbl.set_text("N/A")

        self.rubric_store.clear()
        self.trends_store.clear()
        self.lbl_val_detail.set_markup(f"• Valuation metrics unavailable for {GLib.markup_escape_text(ticker)}.")
        self.quality_narrative_text.get_buffer().set_text(f"Fundamental data is unavailable from provider for ticker {ticker}: {error_msg}")
        self.lbl_trend_summary.set_markup(f"<span color='#888888'>Annual trends unavailable for {GLib.markup_escape_text(ticker)}.</span>")

    def _update_fundamentals_ui(self, fund: Dict[str, Any]):
        self.current_fundamentals = fund
        ev = fund.get("evaluation", {})

        company_name = fund.get("company_name", fund.get("ticker", "Unknown"))
        ticker = fund.get("ticker", "Unknown")
        sector = fund.get("sector", "N/A")
        industry = fund.get("industry", "N/A")
        currency = fund.get("currency", "USD")
        provider = fund.get("data_provider", "Yahoo Finance")
        retrieval = fund.get("retrieval_timestamp", "N/A")

        self.lbl_fund_company.set_markup(
            f"<b>{GLib.markup_escape_text(company_name)} ({GLib.markup_escape_text(ticker)})</b> | "
            f"{GLib.markup_escape_text(sector)} • {GLib.markup_escape_text(industry)} "
            f"<span size='small' color='#666666'>[{GLib.markup_escape_text(provider)}, {GLib.markup_escape_text(currency)}, {GLib.markup_escape_text(retrieval[:10])}]</span>"
        )

        badge_color = ev.get("badge_color", "#333333")
        rating = ev.get("rating", "N/A")
        score = ev.get("score", 0)
        coverage = ev.get("coverage_pct", 0.0)

        if ev.get("score_available"):
            self.lbl_fund_badge.set_markup(f"<span size='large' weight='bold' color='{badge_color}'>● {rating} • Score: {score}/100</span>")
            self.lbl_pillar_a_score.set_markup(f"<span size='medium' weight='bold' color='{badge_color}'>{rating} ({score}/100)</span>")
        else:
            self.lbl_fund_badge.set_markup(f"<span size='large' weight='bold' color='#888888'>● WITHHELD ({rating})</span>")
            self.lbl_pillar_a_score.set_markup("<span size='medium' weight='bold' color='#888888'>Score: WITHHELD</span>")

        self.lbl_pillar_a_sub.set_markup(f"<span size='small' color='#666666'>Coverage: {coverage:.1f}% (Min 60%)</span>")

        pe_str = fund.get("trailing_pe", {}).get("fmt", "—")
        pb_str = fund.get("price_to_book", {}).get("fmt", "—")
        self.lbl_pillar_b_multiples.set_markup(f"<span size='medium' weight='bold'>P/E: {pe_str} | P/B: {pb_str}</span>")
        self.lbl_pillar_b_sub.set_markup(f"<span size='small' color='#666666'>Mkt Cap: {fund.get('market_cap', {}).get('fmt', '—')}</span>")

        self.lbl_val_gross_margin.set_markup(f"<b>{GLib.markup_escape_text(fund.get('gross_margin', {}).get('fmt', '—'))}</b>")
        self.lbl_val_net_margin.set_markup(f"<b>{GLib.markup_escape_text(fund.get('net_margin', {}).get('fmt', '—'))}</b>")
        self.lbl_val_roe.set_markup(f"<b>{GLib.markup_escape_text(fund.get('roe', {}).get('fmt', '—'))}</b>")

        self.lbl_val_current_ratio.set_markup(f"<b>{GLib.markup_escape_text(fund.get('current_ratio', {}).get('fmt', '—'))}</b>")
        self.lbl_val_quick_ratio.set_markup(f"<b>{GLib.markup_escape_text(fund.get('quick_ratio', {}).get('fmt', '—'))}</b>")
        self.lbl_val_de_ratio.set_markup(f"<b>{GLib.markup_escape_text(fund.get('debt_to_equity', {}).get('fmt', '—'))}</b>")

        self.lbl_val_ocf.set_markup(f"<b>{GLib.markup_escape_text(fund.get('operating_cashflow', {}).get('fmt', '—'))}</b>")
        self.lbl_val_fcf.set_markup(f"<b>{GLib.markup_escape_text(fund.get('fcf', {}).get('fmt', '—'))}</b>")
        self.lbl_val_mktcap.set_markup(f"<b>{GLib.markup_escape_text(fund.get('market_cap', {}).get('fmt', '—'))}</b>")

        self.lbl_val_rev_growth.set_markup(f"<b>{GLib.markup_escape_text(fund.get('revenue_growth', {}).get('fmt', '—'))}</b>")
        self.lbl_val_earn_growth.set_markup(f"<b>{GLib.markup_escape_text(fund.get('earnings_growth', {}).get('fmt', '—'))}</b>")
        self.lbl_val_pe.set_markup(f"<b>{GLib.markup_escape_text(fund.get('trailing_pe', {}).get('fmt', '—'))}</b>")

        verdict = ev.get("verdict", "")
        self.lbl_fund_verdict.set_markup(f"<b>Company Quality Verdict:</b> <span color='#222222'>{GLib.markup_escape_text(verdict)}</span>")

        self.rubric_store.clear()
        for item in ev.get("rubric_breakdown", []):
            self.rubric_store.append([
                item.get("category", ""),
                item.get("name", ""),
                item.get("reported", ""),
                f"{item.get('weight', 0):.0f}",
                f"{item.get('points_earned', 0):.1f}",
                item.get("rule", ""),
            ])

        val_text = (
            f"• Market Capitalization : {fund.get('market_cap', {}).get('fmt', 'N/A')}\n"
            f"• Trailing P/E Ratio    : {fund.get('trailing_pe', {}).get('fmt', 'N/A')}\n"
            f"• Forward P/E Ratio     : {fund.get('forward_pe', {}).get('fmt', 'N/A')}\n"
            f"• Price-to-Book (P/B)   : {fund.get('price_to_book', {}).get('fmt', 'N/A')}\n"
            f"• PEG Ratio             : {fund.get('peg_ratio', {}).get('fmt', 'N/A')}"
        )
        self.lbl_val_detail.set_text(val_text)

        narr_lines = [
            f"COMPANY FINANCIAL HEALTH ASSESSMENT: {company_name} ({ticker})",
            f"Sector: {sector} | Industry: {industry} | Currency: {currency}",
            f"Data Coverage: {coverage:.1f}% | Heuristic Score: {score}/100 ({rating} - {ev.get('stance')})",
            "=" * 65,
            f"EXECUTIVE QUALITY VERDICT:\n{verdict}\n",
        ]
        if ev.get("caveat"):
            narr_lines.append(f"CAUTIONARY NOTE:\n{ev.get('caveat')}\n")

        narr_lines.append("=" * 65)
        narr_lines.append("FINANCIAL HEALTH & LIQUIDITY ANALYSIS:")
        narr_lines.append(ev.get("detailed_analysis", "No detailed narrative available."))
        self.quality_narrative_text.get_buffer().set_text("\n".join(narr_lines))

        self._update_trends_ui(fund)

    def _update_trends_ui(self, fund: Optional[Dict[str, Any]]):
        if not fund or "error" in fund:
            self.trends_store.clear()
            self.lbl_trend_summary.set_markup("<span color='#888888'>Annual filing trends unavailable.</span>")
            return

        trends = fund.get("annual_trends", [])
        self.trends_store.clear()
        if not trends:
            self.lbl_trend_summary.set_markup("<span color='#888888'>Multi-period audited statement history not supplied by provider for this symbol.</span>")
            return

        for t in trends:
            self.trends_store.append([
                t.get("period", ""),
                t.get("date", ""),
                t.get("revenue_fmt", "—"),
                t.get("gross_profit_fmt", "—"),
                t.get("gross_margin_fmt", "—"),
                t.get("operating_income_fmt", "—"),
                t.get("operating_margin_fmt", "—"),
                t.get("net_income_fmt", "—"),
                t.get("net_margin_fmt", "—"),
            ])

        if len(trends) >= 2:
            rev_new = trends[0].get("revenue", 0)
            rev_old = trends[-1].get("revenue", 0)
            years = len(trends) - 1
            if rev_new and rev_old and rev_old > 0:
                pct_change = ((rev_new - rev_old) / rev_old) * 100.0
                obs = (
                    f"<b>Multi-Period Trend Analysis:</b> Revenue expanded by {pct_change:+.1f}% across {years} annual periods "
                    f"(from {trends[-1].get('revenue_fmt')} in {trends[-1].get('period')} to {trends[0].get('revenue_fmt')} in {trends[0].get('period')}). "
                    f"Operating margin moved from {trends[-1].get('operating_margin_fmt')} to {trends[0].get('operating_margin_fmt')}."
                )
            else:
                obs = "<b>Multi-Period Trend Analysis:</b> Audited annual figures displayed above. Review margin expansion or contraction."
        else:
            obs = "<b>Multi-Period Trend Analysis:</b> Single annual period available from provider."

        self.lbl_trend_summary.set_markup(obs)

    def _update_technicals_ui(self, analysis: Dict[str, Any]):
        stance = analysis.get("overall_stance", "NEUTRAL")
        badge_color = analysis.get("badge_color", "#333333")
        self.lbl_stance_badge.set_markup(f"<span size='xx-large' weight='bold' color='{badge_color}'>{stance}</span>")

        bulls = analysis.get("bullish_points", 0)
        total = analysis.get("total_checks", 0)
        close = analysis.get("latest_close", 0)
        self.lbl_stance_detail.set_markup(
            f"<b>{bulls} of {total}</b> checks bullish. Latest close: <b>{close:.2f}</b>"
        )
        self.lbl_pillar_c_stance.set_markup(f"<span size='medium' weight='bold' color='{badge_color}'>{stance} ({bulls}/{total})</span>")

        rsi = analysis.get("rsi_14")
        self.lbl_m_rsi.set_text(f"{rsi:.2f} ({analysis.get('rsi_condition', 'N/A')})" if rsi is not None else "N/A")

        macd = analysis.get("macd_line")
        sig = analysis.get("macd_signal")
        self.lbl_m_macd.set_text(f"{macd:.2f} (Sig: {sig:.2f})" if (macd is not None and sig is not None) else "N/A")

        bb_u = analysis.get("bb_upper")
        bb_l = analysis.get("bb_lower")
        self.lbl_m_bb.set_text(f"{bb_l:.2f} - {bb_u:.2f}" if (bb_u is not None and bb_l is not None) else "N/A")

        sma20 = analysis.get("sma_20")
        self.lbl_m_ma.set_text(f"20 SMA: {sma20:.2f}" if sma20 is not None else "N/A")

        stoch_k = analysis.get("stoch_k")
        stoch_d = analysis.get("stoch_d")
        self.lbl_m_stoch.set_text(f"%K={stoch_k:.1f}, %D={stoch_d:.1f}" if (stoch_k is not None and stoch_d is not None) else "N/A")

        fib = analysis.get("fibonacci", {})
        sup = fib.get("nearest_support")
        res = fib.get("nearest_resistance")
        fib_str = f"Sup: {sup[1]:.2f} | Res: {res[1]:.2f}" if (sup and res) else "N/A"
        self.lbl_m_fib.set_text(fib_str)

        self.signals_store.clear()
        for cat, bias, desc in analysis.get("signals", []):
            self.signals_store.append([cat, bias, desc])

        buffer_rep = self.report_text.get_buffer()
        buffer_rep.set_text(analysis.get("synthesis_report", ""))

    def _populate_table(self, rows: List[Dict], include_indicators: bool):
        self.setup_table_columns(include_indicators)
        self.liststore.clear()
        for r in rows:
            if include_indicators:
                self.liststore.append([
                    str(r.get("Date", "")),
                    f"{float(r.get('Close', 0)):.2f}" if r.get("Close") is not None else "",
                    f"{float(r.get('Open', 0)):.2f}" if r.get("Open") is not None else "",
                    f"{float(r.get('High', 0)):.2f}" if r.get("High") is not None else "",
                    f"{float(r.get('Low', 0)):.2f}" if r.get("Low") is not None else "",
                    f"{int(r.get('Volume', 0)):,}" if r.get("Volume") is not None else "",
                    str(r.get("SMA_20", "")),
                    str(r.get("RSI_14", "")),
                    str(r.get("BB_Upper", "")),
                    str(r.get("BB_Lower", "")),
                    str(r.get("MACD", "")),
                    str(r.get("Stoch_K", "")),
                ])
            else:
                self.liststore.append([
                    str(r.get("Date", "")),
                    f"{float(r.get('Open', 0)):.2f}" if r.get("Open") is not None else "",
                    f"{float(r.get('High', 0)):.2f}" if r.get("High") is not None else "",
                    f"{float(r.get('Low', 0)):.2f}" if r.get("Low") is not None else "",
                    f"{float(r.get('Close', 0)):.2f}" if r.get("Close") is not None else "",
                    f"{float(r.get('Adj Close', 0)):.2f}" if r.get("Adj Close") is not None else "",
                    f"{int(r.get('Volume', 0)):,}" if r.get("Volume") is not None else "",
                ])

    def on_prompt_selection_changed(self, widget):
        selected_id = self.combo_prompts.get_active_id() or "synthesis"
        buf = self.prompt_text_view.get_buffer()

        ticker = self.current_ticker or self.entry_ticker.get_text().strip().upper() or "TICKER"
        fund = self.current_fundamentals or {}
        analysis = self.current_analysis or {}

        fund_prompts = fund.get("ai_prompts", {})
        analysis_prompts = analysis.get("ai_prompts", {})

        if selected_id in fund_prompts:
            prompt_content = fund_prompts[selected_id]
        elif selected_id in analysis_prompts:
            prompt_content = analysis_prompts[selected_id]
        else:
            prompt_content = "Download and analyze data for a ticker to generate this prompt."

        buf.set_text(prompt_content)

    def on_copy_prompt_clicked(self, widget):
        buf = self.prompt_text_view.get_buffer()
        start = buf.get_start_iter()
        end = buf.get_end_iter()
        text = buf.get_text(start, end, True)
        if text.strip():
            cb = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
            cb.set_text(text, -1)
            self.status_label.set_markup("<span color='#0e8a16'>📋 Prompt copied to clipboard!</span>")

    def on_open_file_clicked(self, widget):
        if self.last_saved_file and os.path.exists(self.last_saved_file):
            try:
                subprocess.Popen(["xdg-open", self.last_saved_file])
            except Exception as e:
                self.show_error(f"Could not open file: {e}")

    def on_open_folder_clicked(self, widget):
        if self.last_saved_file and os.path.exists(self.last_saved_file):
            folder = os.path.dirname(os.path.abspath(self.last_saved_file))
            try:
                subprocess.Popen(["xdg-open", folder])
            except Exception as e:
                self.show_error(f"Could not open folder: {e}")

    def show_error(self, message: str):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Notice",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()


def main():
    app = StockDownloaderApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    app.spinner.stop()
    Gtk.main()


if __name__ == "__main__":
    main()
