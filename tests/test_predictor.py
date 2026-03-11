"""
Tests for src/predictor.py

PatternResult dataclass・detect_pattern()・draw_prediction() を検証する。
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.predictor import PatternResult, detect_pattern, draw_prediction


# ---------------------------------------------------------------------------
# テスト用 DataFrame ヘルパー
# ---------------------------------------------------------------------------

def _make_df(close: np.ndarray) -> pd.DataFrame:
    """Close 系列から最小限の OHLCV DataFrame を作る。"""
    n = len(close)
    index = pd.date_range(
        "2026-02-20 09:00", periods=n, freq="1min", tz="Asia/Tokyo"
    )
    return pd.DataFrame(
        {
            "Open":   close,
            "High":   close * 1.002,
            "Low":    close * 0.998,
            "Close":  close,
            "Volume": np.full(n, 100000),
        },
        index=index,
    )


def _make_monotone_df(n: int = 30) -> pd.DataFrame:
    """単調増加の Close 系列（パターンなし）。"""
    return _make_df(np.linspace(100.0, 115.0, n))


def _make_double_top_df() -> pd.DataFrame:
    """ダブルトップ形状: 2 つの近似価格の山（差 ≤ 2%）と間に谷。n=60"""
    n = 60
    close = np.full(n, 100.0)
    # Peak 1 at index 10
    close[5:16] = [100, 102, 104, 108, 112, 115.0, 112, 108, 104, 102, 100]
    # Valley at index 25
    close[20:31] = [100, 98, 96, 93, 91, 90.0, 91, 93, 96, 98, 100]
    # Peak 2 at index 45 (within 2% of peak 1: 114.5/115 = 99.6%)
    close[40:51] = [100, 102, 104, 108, 112, 114.5, 112, 108, 104, 102, 100]
    return _make_df(close)


def _make_double_bottom_df() -> pd.DataFrame:
    """ダブルボトム形状: 2 つの近似価格の谷（差 ≤ 2%）と間に山。n=60"""
    n = 60
    close = np.full(n, 100.0)
    # Trough 1 at index 10
    close[5:16] = [100, 98, 96, 92, 88, 85.0, 88, 92, 96, 98, 100]
    # Peak at index 25
    close[20:31] = [100, 102, 104, 107, 109, 110.0, 109, 107, 104, 102, 100]
    # Trough 2 at index 45 (within 2% of trough 1: 85.5/85 = 100.6%)
    close[40:51] = [100, 98, 96, 92, 88, 85.5, 88, 92, 96, 98, 100]
    return _make_df(close)


def _make_head_and_shoulders_df() -> pd.DataFrame:
    """ヘッドアンドショルダー: 左右の肩 + 中央の頭（最高値）。n=80"""
    n = 80
    close = np.full(n, 100.0)
    # Left shoulder at index 10
    close[5:16]  = [100, 102, 104, 106, 107, 108.0, 107, 106, 104, 102, 100]
    # Head at index 35 (highest)
    close[30:41] = [100, 103, 107, 112, 118, 120.0, 118, 112, 107, 103, 100]
    # Right shoulder at index 65 (within 3% of left shoulder: 108.5/108 = 100.5%)
    close[60:71] = [100, 102, 104, 106, 107, 108.5, 107, 106, 104, 102, 100]
    return _make_df(close)


def _make_inverse_head_and_shoulders_df() -> pd.DataFrame:
    """逆ヘッドアンドショルダー: 左右の肩 + 中央の頭（最低値）。n=80"""
    n = 80
    close = np.full(n, 100.0)
    # Left shoulder at index 10
    close[5:16]  = [100, 98, 96, 94, 93, 92.0, 93, 94, 96, 98, 100]
    # Head at index 35 (lowest)
    close[30:41] = [100, 97, 93, 88, 82, 80.0, 82, 88, 93, 97, 100]
    # Right shoulder at index 65 (within 3% of left shoulder: 92.5/92 = 100.5%)
    close[60:71] = [100, 98, 96, 94, 93, 92.5, 93, 94, 96, 98, 100]
    return _make_df(close)


def _make_pattern_result() -> PatternResult:
    """テスト用の PatternResult インスタンスを作る。"""
    index = pd.date_range(
        "2026-02-20 10:00", periods=10, freq="1min", tz="Asia/Tokyo"
    )
    return PatternResult(
        name="ダブルトップ",
        direction="down",
        forecast_prices=[100.0 - i * 0.5 for i in range(10)],
        forecast_index=index,
    )


# ---------------------------------------------------------------------------
# detect_pattern() のテスト
# ---------------------------------------------------------------------------

class TestDetectPattern:
    """detect_pattern() がパターンを正しく検出するケース"""

    def test_returns_none_when_data_less_than_30(self):
        df = _make_df(np.full(29, 100.0))
        assert detect_pattern(df) is None

    def test_returns_none_when_no_pattern_found(self):
        df = _make_monotone_df(30)
        assert detect_pattern(df) is None

    def test_detects_double_top(self):
        result = detect_pattern(_make_double_top_df())
        assert result is not None
        assert result.name == "ダブルトップ"
        assert result.direction == "down"

    def test_detects_double_bottom(self):
        result = detect_pattern(_make_double_bottom_df())
        assert result is not None
        assert result.name == "ダブルボトム"
        assert result.direction == "up"

    def test_detects_head_and_shoulders(self):
        result = detect_pattern(_make_head_and_shoulders_df())
        assert result is not None
        assert result.name == "ヘッドアンドショルダー"
        assert result.direction == "down"

    def test_detects_inverse_head_and_shoulders(self):
        result = detect_pattern(_make_inverse_head_and_shoulders_df())
        assert result is not None
        assert result.name == "逆ヘッドアンドショルダー"
        assert result.direction == "up"

    def test_result_has_10_forecast_prices(self):
        result = detect_pattern(_make_double_top_df())
        assert result is not None
        assert len(result.forecast_prices) == 10

    def test_result_forecast_index_length(self):
        result = detect_pattern(_make_double_top_df())
        assert result is not None
        assert len(result.forecast_index) == 10

    def test_result_forecast_index_starts_after_last_ts(self):
        df = _make_double_top_df()
        result = detect_pattern(df)
        assert result is not None
        assert result.forecast_index[0] > df.index[-1]

    def test_down_direction_forecast_decreases(self):
        result = detect_pattern(_make_double_top_df())
        assert result is not None
        assert result.direction == "down"
        assert result.forecast_prices[-1] < result.forecast_prices[0]

    def test_up_direction_forecast_increases(self):
        result = detect_pattern(_make_double_bottom_df())
        assert result is not None
        assert result.direction == "up"
        assert result.forecast_prices[-1] > result.forecast_prices[0]


# ---------------------------------------------------------------------------
# draw_prediction() のテスト
# ---------------------------------------------------------------------------

class TestDrawPrediction:
    """draw_prediction() が axes に正しく描画するケース"""

    def test_draw_prediction_calls_ax_plot(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        assert ax.plot.called

    def test_draw_prediction_uses_dashed_line(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        _, kwargs = ax.plot.call_args
        assert kwargs.get("linestyle") == "--"

    def test_draw_prediction_uses_integer_x_values(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        x_values = ax.plot.call_args[0][0]
        assert x_values == list(range(60, 70))

    def test_draw_prediction_calls_ax_text_for_pattern_name(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        assert ax.text.called

    def test_draw_prediction_pattern_name_in_ax_text(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        text_arg = ax.text.call_args[0][2]
        assert text_arg == "ダブルトップ"
