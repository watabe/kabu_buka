"""
Tests for main.py の --predict フラグ
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from main import main
from src.models import StockInfo


def _make_stock_info() -> StockInfo:
    return StockInfo(
        code="7203",
        ticker="7203.T",
        name="トヨタ自動車",
        current_price=3410.0,
        previous_close=3392.0,
        open_price=3400.0,
        day_high=3420.0,
        day_low=3390.0,
        volume=560000,
        market_cap=55_000_000_000_000,
    )


def _make_ohlcv(n: int = 3) -> pd.DataFrame:
    index = pd.date_range(
        "2026-02-20 09:00", periods=n, freq="1min", tz="Asia/Tokyo"
    )
    return pd.DataFrame(
        {
            "Open":   [3400.0, 3410.0, 3405.0][:n],
            "High":   [3415.0, 3420.0, 3418.0][:n],
            "Low":    [3395.0, 3405.0, 3400.0][:n],
            "Close":  [3410.0, 3405.0, 3412.0][:n],
            "Volume": [100000, 120000, 95000][:n],
        },
        index=index,
    )


class TestMainPredictFlag:
    """main.py の --predict フラグが正しく CLI に追加されているケース"""

    @patch("main.print_stock_info")
    @patch("main.fetch_stock_info")
    def test_predict_flag_is_accepted(self, mock_fetch_stock, mock_print):
        """--predict 引数が argparse に登録されており、エラーなく受け付けられること"""
        mock_fetch_stock.return_value = _make_stock_info()
        with patch("sys.argv", ["main.py", "7203", "--predict"]):
            result = main()
        assert result == 0

    @patch("src.chart.plot_intraday")
    @patch("main.fetch_intraday_data")
    @patch("main.print_stock_info")
    @patch("main.fetch_stock_info")
    def test_predict_passed_to_plot_intraday(
        self, mock_fetch_stock, mock_print, mock_fetch_data, mock_plot_intraday
    ):
        """--chart --predict 時に plot_intraday へ predict=True が渡されること"""
        mock_fetch_stock.return_value = _make_stock_info()
        mock_fetch_data.return_value = _make_ohlcv()
        with patch("sys.argv", ["main.py", "7203", "--chart", "--predict"]):
            main()
        _, kwargs = mock_plot_intraday.call_args
        assert kwargs.get("predict") is True

    @patch("src.chart.plot_intraday_live")
    @patch("main.print_stock_info")
    @patch("main.fetch_stock_info")
    def test_predict_passed_to_plot_intraday_live(
        self, mock_fetch_stock, mock_print, mock_plot_live
    ):
        """--chart --interval 60 --predict 時に plot_intraday_live へ predict=True が渡されること"""
        mock_fetch_stock.return_value = _make_stock_info()
        with patch("sys.argv", ["main.py", "7203", "--chart", "--interval", "60", "--predict"]):
            main()
        _, kwargs = mock_plot_live.call_args
        assert kwargs.get("predict") is True
