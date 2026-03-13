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


def _make_triple_top_df() -> pd.DataFrame:
    """トリプルトップ: 3山の価格がほぼ同値（差≤3%）、谷の深さも均一。n=90

    Peak1@10=115.0, Peak2@45=114.5(中央が最高でない→H&S不成立), Peak3@80=115.2
    Valley1@25=88.0, Valley2@65=88.5 (横ばい→上昇三角形と区別)
    """
    n = 90
    kx = [0,  5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 65, 70, 80, 85, 89]
    ky = [100,107,115,107,100, 88, 95,107,114.5,107,100,88.5, 95,115.2,107,100]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_triple_bottom_df() -> pd.DataFrame:
    """トリプルボトム: 3谷の価格がほぼ同値（差≤3%）、山の高さも均一。n=90

    Trough1@10=85.0, Trough2@45=85.5(中央が最低でない→逆H&S不成立), Trough3@80=85.2
    Peak1@25=112.0, Peak2@65=112.5
    """
    n = 90
    kx = [0,  5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 65, 70, 80, 85, 89]
    ky = [100, 93, 85, 93,100,112,105, 93, 85.5, 93,100,112.5,105,85.2, 93,100]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_ascending_triangle_df() -> pd.DataFrame:
    """上昇三角形: 水平抵抗（峰≈115）＋上昇支持（谷が88→103に切り上がり）。n=60

    5つの極大がほぼ水平（傾き≈0）、4つの極小が上昇（傾き>0）。
    谷が上昇しているのでトリプルトップと区別できる。
    """
    n = 60
    kx = [0,  5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    ky = [102,115.0, 88.0,115.1, 93.0,114.9, 98.0,115.0,103.0,115.2,108]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_descending_triangle_df() -> pd.DataFrame:
    """下降三角形: 下降抵抗（峰が115→103に切り下がり）＋水平支持（谷≈88）。n=60

    5つの極大が下降（傾き<0）、4つの極小がほぼ水平（傾き≈0）。
    """
    n = 60
    kx = [0,  5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    ky = [102,115.0, 88.0,112.0, 88.0,109.0, 88.0,106.0, 88.0,103.0, 88.0]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_symmetrical_triangle_df() -> pd.DataFrame:
    """対称三角形: 下降抵抗（峰115→107）＋上昇支持（谷88→97）の収束。n=60

    直前の全体トレンド上昇 → direction="up" を期待。
    """
    n = 60
    kx = [0,  5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    ky = [88,115.0, 88.0,113.0, 91.0,111.0, 94.0,109.0, 97.0,107.0,101]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_rising_wedge_df() -> pd.DataFrame:
    """ライジングウェッジ: 峰も谷も上昇、谷の傾きが峰より急（収束）。n=60

    峰: 107→115（傾き≈+0.17/bar）、谷: 96→112（傾き≈+0.33/bar）
    → slope_low > slope_high → 収束 → direction="down"
    """
    n = 60
    kx = [0,  5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    ky = [100,107.0, 96.0,109.0, 99.0,111.0,103.0,113.0,107.0,115.0,112]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_falling_wedge_df() -> pd.DataFrame:
    """フォーリングウェッジ: 峰も谷も下落、峰の傾きが谷より急（収束）。n=60

    峰: 115→103（傾き≈-0.25/bar）、谷: 90→83（傾き≈-0.15/bar）
    → slope_high < slope_low → 収束 → direction="up"
    """
    n = 60
    kx = [0,  5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
    ky = [105,115.0, 90.0,112.0, 88.0,109.0, 86.5,106.0, 84.5,103.0, 83]
    close = np.interp(np.arange(n), kx, ky)
    return _make_df(close)


def _make_bullish_flag_df() -> pd.DataFrame:
    """強気フラッグ: 急騰ポール（+30%）＋小幅下落チャンネル（-5%未満）。n=60"""
    n = 60
    pole  = np.linspace(100.0, 130.0, 20)
    flag  = np.linspace(130.0, 124.0, 40) + np.sin(np.linspace(0, 4 * np.pi, 40)) * 0.5
    return _make_df(np.concatenate([pole, flag]))


def _make_bearish_flag_df() -> pd.DataFrame:
    """弱気フラッグ: 急落ポール（-23%）＋小幅上昇チャンネル（+6%未満）。n=60"""
    n = 60
    pole  = np.linspace(130.0, 100.0, 20)
    flag  = np.linspace(100.0, 106.0, 40) + np.sin(np.linspace(0, 4 * np.pi, 40)) * 0.5
    return _make_df(np.concatenate([pole, flag]))


def _make_cup_and_handle_df() -> pd.DataFrame:
    """カップアンドハンドル: U字カップ＋右端の小幅ハンドル下落。n=90"""
    left   = np.linspace(115.0,  90.0, 40)
    bottom = np.full(10, 90.0)
    right  = np.linspace( 90.0, 114.0, 30)
    handle = np.array([114.0,113.5,113.0,112.5,112.0,112.5,113.0,113.5,114.0,114.5])
    return _make_df(np.concatenate([left, bottom, right, handle]))


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
        confidence=0.8,
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

    def test_detects_triple_top(self):
        result = detect_pattern(_make_triple_top_df())
        assert result is not None
        assert result.name == "トリプルトップ"
        assert result.direction == "down"

    def test_detects_triple_bottom(self):
        result = detect_pattern(_make_triple_bottom_df())
        assert result is not None
        assert result.name == "トリプルボトム"
        assert result.direction == "up"

    def test_detects_ascending_triangle(self):
        result = detect_pattern(_make_ascending_triangle_df())
        assert result is not None
        assert result.name == "上昇三角形"
        assert result.direction == "up"

    def test_detects_descending_triangle(self):
        result = detect_pattern(_make_descending_triangle_df())
        assert result is not None
        assert result.name == "下降三角形"
        assert result.direction == "down"

    def test_detects_symmetrical_triangle(self):
        result = detect_pattern(_make_symmetrical_triangle_df())
        assert result is not None
        assert result.name == "対称三角形"

    def test_detects_rising_wedge(self):
        result = detect_pattern(_make_rising_wedge_df())
        assert result is not None
        assert result.name == "ライジングウェッジ"
        assert result.direction == "down"

    def test_detects_falling_wedge(self):
        result = detect_pattern(_make_falling_wedge_df())
        assert result is not None
        assert result.name == "フォーリングウェッジ"
        assert result.direction == "up"

    def test_detects_bullish_flag(self):
        result = detect_pattern(_make_bullish_flag_df())
        assert result is not None
        assert result.name == "強気フラッグ"
        assert result.direction == "up"

    def test_detects_bearish_flag(self):
        result = detect_pattern(_make_bearish_flag_df())
        assert result is not None
        assert result.name == "弱気フラッグ"
        assert result.direction == "down"

    def test_detects_cup_and_handle(self):
        result = detect_pattern(_make_cup_and_handle_df())
        assert result is not None
        assert result.name == "カップアンドハンドル"
        assert result.direction == "up"

    def test_new_patterns_return_10_forecast_prices(self):
        for df_fn in [
            _make_triple_top_df, _make_triple_bottom_df,
            _make_ascending_triangle_df, _make_descending_triangle_df,
            _make_rising_wedge_df, _make_falling_wedge_df,
        ]:
            result = detect_pattern(df_fn())
            if result is not None:
                assert len(result.forecast_prices) == 10

    def test_result_has_confidence_field(self):
        result = detect_pattern(_make_double_top_df())
        assert result is not None
        assert hasattr(result, "confidence")

    def test_confidence_is_between_0_and_1(self):
        pattern_fns = [
            _make_double_top_df, _make_double_bottom_df,
            _make_head_and_shoulders_df, _make_inverse_head_and_shoulders_df,
            _make_triple_top_df, _make_triple_bottom_df,
            _make_ascending_triangle_df, _make_descending_triangle_df,
            _make_symmetrical_triangle_df,
            _make_rising_wedge_df, _make_falling_wedge_df,
            _make_bullish_flag_df, _make_bearish_flag_df,
            _make_cup_and_handle_df,
        ]
        for fn in pattern_fns:
            result = detect_pattern(fn())
            assert result is not None, f"{fn.__name__} returned None"
            assert 0.0 <= result.confidence <= 1.0, (
                f"{fn.__name__}: confidence={result.confidence} out of range"
            )

    def test_double_top_target_below_neckline(self):
        df = _make_double_top_df()
        result = detect_pattern(df)
        assert result is not None
        # ネックライン（谷の最安値）は約 90。目標はネックライン以下になること
        neckline_approx = 90.0
        assert result.forecast_prices[-1] < neckline_approx

    def test_double_bottom_target_above_neckline(self):
        df = _make_double_bottom_df()
        result = detect_pattern(df)
        assert result is not None
        # ネックライン（山の最高値）は約 110。目標はネックライン以上になること
        neckline_approx = 110.0
        assert result.forecast_prices[-1] > neckline_approx

    def test_head_and_shoulders_target_below_neckline(self):
        df = _make_head_and_shoulders_df()
        result = detect_pattern(df)
        assert result is not None
        # ネックラインは肩間の谷 ≈ 100.0。目標はそれより低くなること
        assert result.forecast_prices[-1] < 100.0

    def test_bullish_flag_target_above_last_close(self):
        df = _make_bullish_flag_df()
        result = detect_pattern(df)
        assert result is not None
        # 強気フラッグ: 目標 = last_close + pole_height (pole は +30%)
        # last_close ≈ 124, pole_height ≈ 30 → target ≈ 154
        assert result.forecast_prices[-1] > float(df["Close"].iloc[-1])

    def test_bearish_flag_target_below_last_close(self):
        df = _make_bearish_flag_df()
        result = detect_pattern(df)
        assert result is not None
        # 弱気フラッグ: 目標 = last_close - pole_height (pole は -30)
        assert result.forecast_prices[-1] < float(df["Close"].iloc[-1])

    def test_forecast_is_linear_from_last_close_to_target(self):
        """forecast_prices が last_close から target まで線形補間になっていること。"""
        df = _make_double_top_df()
        result = detect_pattern(df)
        assert result is not None
        prices = result.forecast_prices
        # 隣接する差がほぼ等間隔（等差数列）
        diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        assert len(diffs) > 0
        # 全差が同じ符号かつ絶対値のばらつきが小さい
        assert all(d < 0 for d in diffs), "down パターンなので差はすべて負"
        max_diff = max(abs(d) for d in diffs)
        min_diff = min(abs(d) for d in diffs)
        assert max_diff - min_diff < 0.01, f"等差数列でない: diffs={diffs}"

    def test_higher_confidence_for_tighter_double_top(self):
        """山の差が小さいほど信頼度が高いことを確認（2データセット比較）。"""
        # tight: 山の差 0.1%
        n = 60
        close_tight = np.full(n, 100.0)
        close_tight[5:16]  = [100,102,104,108,112,115.0,112,108,104,102,100]
        close_tight[20:31] = [100, 98, 96, 93, 91, 90.0, 91, 93, 96, 98,100]
        close_tight[40:51] = [100,102,104,108,112,115.1,112,108,104,102,100]  # 差 0.09%
        result_tight = detect_pattern(_make_df(close_tight))

        # loose: 山の差 1.8%
        close_loose = np.full(n, 100.0)
        close_loose[5:16]  = [100,102,104,108,112,115.0,112,108,104,102,100]
        close_loose[20:31] = [100, 98, 96, 93, 91, 90.0, 91, 93, 96, 98,100]
        close_loose[40:51] = [100,102,104,108,112,112.9,112,108,104,102,100]  # 差 1.8%
        result_loose = detect_pattern(_make_df(close_loose))

        assert result_tight is not None and result_loose is not None
        assert result_tight.name == "ダブルトップ" and result_loose.name == "ダブルトップ"
        assert result_tight.confidence > result_loose.confidence

    def test_new_patterns_do_not_trigger_on_existing_pattern_data(self):
        """既存パターン用データで新パターン名が返らないこと（後退テスト）。"""
        new_names = {
            "トリプルトップ", "トリプルボトム",
            "上昇三角形", "下降三角形", "対称三角形",
            "ライジングウェッジ", "フォーリングウェッジ",
            "強気フラッグ", "弱気フラッグ", "カップアンドハンドル",
        }
        for df_fn, expected in [
            (_make_double_top_df,               "ダブルトップ"),
            (_make_double_bottom_df,            "ダブルボトム"),
            (_make_head_and_shoulders_df,       "ヘッドアンドショルダー"),
            (_make_inverse_head_and_shoulders_df, "逆ヘッドアンドショルダー"),
        ]:
            result = detect_pattern(df_fn())
            assert result is not None
            assert result.name == expected, (
                f"{df_fn.__name__} returned '{result.name}', expected '{expected}'"
            )


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
        assert text_arg == "ダブルトップ (80%)"

    def test_draw_prediction_shows_confidence_in_label(self):
        ax = MagicMock()
        draw_prediction(ax, _make_pattern_result(), n_existing=60)
        text_arg = ax.text.call_args[0][2]
        assert "(" in text_arg and "%" in text_arg
