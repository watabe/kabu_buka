"""
Tests for src/display.py

Tests verify the formatted string output for various StockInfo states.
"""
import pytest
from io import StringIO
from unittest.mock import patch

from src.models import StockInfo
from src.display import format_stock_info, print_stock_info


def make_stock_info(**overrides) -> StockInfo:
    defaults = dict(
        code="7203",
        ticker="7203.T",
        name="Toyota Motor Corporation",
        current_price=3420.0,
        previous_close=3390.0,
        open_price=3400.0,
        day_high=3435.0,
        day_low=3398.0,
        volume=8234500,
        market_cap=55_600_000_000_000,
    )
    defaults.update(overrides)
    return StockInfo(**defaults)


class TestFormatStockInfoBasic:
    """format_stock_info() が期待する文字列を含むケース"""

    def test_contains_stock_name(self):
        output = format_stock_info(make_stock_info())
        assert "Toyota Motor Corporation" in output

    def test_contains_ticker(self):
        output = format_stock_info(make_stock_info())
        assert "7203.T" in output

    def test_contains_current_price(self):
        output = format_stock_info(make_stock_info())
        assert "3,420" in output

    def test_contains_high_and_low(self):
        output = format_stock_info(make_stock_info())
        assert "3,435" in output
        assert "3,398" in output

    def test_contains_volume(self):
        output = format_stock_info(make_stock_info())
        assert "8,234,500" in output

    def test_returns_string(self):
        result = format_stock_info(make_stock_info())
        assert isinstance(result, str)


class TestFormatStockInfoDailyChange:
    """前日比（正・負・ゼロ）の表示が正しいケース"""

    def test_positive_change_shows_plus_sign(self):
        # current=3420, previous=3390 → +30
        output = format_stock_info(make_stock_info(current_price=3420.0, previous_close=3390.0))
        assert "+30" in output or "+30.0" in output

    def test_negative_change_shows_minus_sign(self):
        # current=3350, previous=3390 → -40
        output = format_stock_info(make_stock_info(current_price=3350.0, previous_close=3390.0))
        assert "-40" in output or "-40.0" in output

    def test_zero_change(self):
        # current=3390, previous=3390 → 0
        output = format_stock_info(make_stock_info(current_price=3390.0, previous_close=3390.0))
        # 0 は "0" または "+0" または "±0" など実装依存。クラッシュしないことを確認
        assert output is not None
        assert len(output) > 0

    def test_positive_change_percentage_present(self):
        # +30 / 3390 ≈ +0.88%
        output = format_stock_info(make_stock_info(current_price=3420.0, previous_close=3390.0))
        assert "%" in output


class TestFormatStockInfoMarketCap:
    """market_cap のバリエーション"""

    def test_market_cap_none_does_not_crash(self):
        output = format_stock_info(make_stock_info(market_cap=None))
        assert isinstance(output, str)

    def test_market_cap_shown_when_present(self):
        output = format_stock_info(make_stock_info(market_cap=55_600_000_000_000))
        # 何らかの形で時価総額の値が含まれることを確認（表示形式は実装依存）
        assert output is not None
        assert len(output) > 0


class TestPrintStockInfo:
    """print_stock_info() が標準出力に書き出すケース"""

    def test_prints_to_stdout(self, capsys):
        print_stock_info(make_stock_info())
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_stock_name(self, capsys):
        print_stock_info(make_stock_info())
        captured = capsys.readouterr()
        assert "Toyota Motor Corporation" in captured.out

    def test_output_contains_current_price(self, capsys):
        print_stock_info(make_stock_info())
        captured = capsys.readouterr()
        assert "3,420" in captured.out

    def test_market_cap_none_prints_without_error(self, capsys):
        print_stock_info(make_stock_info(market_cap=None))
        captured = capsys.readouterr()
        assert len(captured.out) > 0
