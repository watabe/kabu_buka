"""
Tests for src/chart.py

mplfinance.plot をモック化して GUI を起動せずに検証する。
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call
from pandas import DatetimeIndex

from src.chart import (
    plot_intraday,
    plot_intraday_live,
    _resolve_japanese_font,
    _build_info_text,
    _build_info_parts,
    _JP_FONT_CANDIDATES,
)
from src.models import StockInfo


def _make_ohlcv(n: int = 3) -> pd.DataFrame:
    """テスト用の最小 OHLCV DataFrame を作る。"""
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


def _make_stock_info() -> StockInfo:
    """テスト用の StockInfo インスタンスを作る。"""
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


class TestPlotIntradayCallsMpfPlot:
    """plot_intraday() が mpf.plot を正しく呼び出すケース"""

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_mpf_plot_is_called(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_plot.called

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_mpf_plot_called_once(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_plot.call_count == 1

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_chart_type_is_candle(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_plot.call_args
        assert kwargs.get("type") == "candle"

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_volume_subplot_enabled(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_plot.call_args
        assert kwargs.get("volume") is True

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_title_contains_ticker(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_plot.call_args
        assert kwargs.get("title") == ""


class TestPlotIntradayDataframe:
    """mpf.plot に渡される DataFrame の内容検証"""

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_dataframe_has_required_columns(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        passed_df = mock_plot.call_args[0][0]
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in passed_df.columns

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_dataframe_index_is_datetime(self, mock_plot, mock_show):
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        passed_df = mock_plot.call_args[0][0]
        assert isinstance(passed_df.index, DatetimeIndex)


class TestPlotIntradayEmptyDataframe:
    """空 DataFrame を渡したとき ValueError が発生するケース"""

    def test_raises_value_error_on_empty_df(self):
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )
        with pytest.raises(ValueError):
            plot_intraday(empty_df, info=_make_stock_info())


# ---------------------------------------------------------------------------
# _resolve_japanese_font() のテスト
# ---------------------------------------------------------------------------

def _make_font_entry(name: str) -> MagicMock:
    """fm.fontManager.ttflist のエントリを模倣する MagicMock を作る。"""
    entry = MagicMock()
    entry.name = name
    return entry


class TestResolveJapaneseFont:
    """_resolve_japanese_font() がフォントリストから正しく選択するケース"""

    @patch("src.chart.fm.fontManager")
    def test_returns_hiragino_sans_when_available(self, mock_fm):
        mock_fm.ttflist = [_make_font_entry("Hiragino Sans")]

        result = _resolve_japanese_font()

        assert result == "Hiragino Sans"

    @patch("src.chart.fm.fontManager")
    def test_returns_first_candidate_in_priority_order(self, mock_fm):
        # リストに2番目と3番目の候補しかない場合、先に見つかった方（優先リスト順）を返す
        mock_fm.ttflist = [
            _make_font_entry("Hiragino Maru Gothic Pro"),
            _make_font_entry("Hiragino Kaku Gothic Pro"),
        ]

        result = _resolve_japanese_font()

        assert result == "Hiragino Kaku Gothic Pro"

    @patch("src.chart.fm.fontManager")
    def test_returns_none_when_no_candidate_available(self, mock_fm):
        mock_fm.ttflist = [_make_font_entry("DejaVu Sans")]

        result = _resolve_japanese_font()

        assert result is None

    @patch("src.chart.fm.fontManager")
    def test_returns_none_when_font_list_is_empty(self, mock_fm):
        mock_fm.ttflist = []

        result = _resolve_japanese_font()

        assert result is None

    def test_jp_font_candidates_is_not_empty(self):
        assert len(_JP_FONT_CANDIDATES) > 0

    def test_hiragino_sans_is_first_candidate(self):
        assert _JP_FONT_CANDIDATES[0] == "Hiragino Sans"


# ---------------------------------------------------------------------------
# plot_intraday() 内の make_mpf_style 呼び出し検証
# ---------------------------------------------------------------------------

class TestPlotIntradayFontSetting:
    """plot_intraday() が make_mpf_style に正しい rc を渡すケース"""

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.mpf.make_mpf_style")
    @patch("src.chart._resolve_japanese_font", return_value="Hiragino Sans")
    def test_make_mpf_style_is_called(self, mock_font, mock_make_style, mock_plot, mock_show):
        mock_make_style.return_value = MagicMock()
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_make_style.called

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.mpf.make_mpf_style")
    @patch("src.chart._resolve_japanese_font", return_value="Hiragino Sans")
    def test_rc_contains_unicode_minus_false(self, mock_font, mock_make_style, mock_plot, mock_show):
        mock_make_style.return_value = MagicMock()
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_make_style.call_args
        assert kwargs["rc"]["axes.unicode_minus"] is False

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.mpf.make_mpf_style")
    @patch("src.chart._resolve_japanese_font", return_value="Hiragino Sans")
    def test_rc_contains_font_family_when_font_found(self, mock_font, mock_make_style, mock_plot, mock_show):
        mock_make_style.return_value = MagicMock()
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_make_style.call_args
        assert kwargs["rc"]["font.family"] == "Hiragino Sans"

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.mpf.make_mpf_style")
    @patch("src.chart._resolve_japanese_font", return_value=None)
    def test_rc_has_no_font_family_when_no_font_found(self, mock_font, mock_make_style, mock_plot, mock_show):
        mock_make_style.return_value = MagicMock()
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_make_style.call_args
        assert "font.family" not in kwargs["rc"]

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.mpf.make_mpf_style")
    @patch("src.chart._resolve_japanese_font", return_value="Hiragino Sans")
    def test_style_object_is_passed_to_mpf_plot(self, mock_font, mock_make_style, mock_plot, mock_show):
        style_obj = MagicMock(name="style_object")
        mock_make_style.return_value = style_obj
        mock_plot.return_value = (MagicMock(), MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_plot.call_args
        assert kwargs.get("style") is style_obj


# ---------------------------------------------------------------------------
# plot_intraday() の会社情報テキスト表示検証
# ---------------------------------------------------------------------------

class TestPlotIntradayInfoText:
    """plot_intraday() がチャート下部に会社情報テキストを描画するケース"""

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_mpf_plot_called_with_returnfig_true(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        _, kwargs = mock_plot.call_args
        assert kwargs.get("returnfig") is True

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_fig_subplots_adjust_is_called(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        mock_fig.subplots_adjust.assert_called_once()

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_fig_text_is_called(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_fig.text.call_count >= 4

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_fig_text_contains_company_name_and_ticker(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        all_texts = " ".join(c[0][2] for c in mock_fig.text.call_args_list)
        assert "トヨタ自動車" in all_texts
        assert "7203.T" in all_texts

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_fig_text_contains_current_price(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        all_texts = " ".join(c[0][2] for c in mock_fig.text.call_args_list)
        assert "3,410" in all_texts

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_fig_text_contains_timestamp(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        df = _make_ohlcv()
        plot_intraday(df, info=_make_stock_info())
        all_texts = " ".join(c[0][2] for c in mock_fig.text.call_args_list)
        expected_date = df.index[-1].strftime("%Y-%m-%d")
        assert expected_date in all_texts

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_mpf_show_is_called(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_show.called

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_period_label_in_row1_text(self, mock_plot, mock_show):
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, MagicMock())
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        all_texts = " ".join(c[0][2] for c in mock_fig.text.call_args_list)
        assert "1日 (1分足)" in all_texts


# ---------------------------------------------------------------------------
# _build_info_text() のテスト
# ---------------------------------------------------------------------------

class TestBuildInfoText:
    """_build_info_text() が正しい情報テキストを生成するケース"""

    def test_contains_company_name_and_ticker(self):
        result = _build_info_text(_make_ohlcv(), _make_stock_info())
        assert "トヨタ自動車" in result
        assert "7203.T" in result

    def test_contains_current_price_and_change(self):
        result = _build_info_text(_make_ohlcv(), _make_stock_info())
        assert "3,410" in result   # 現在値
        expected_date = _make_ohlcv().index[-1].strftime("%Y-%m-%d")
        assert expected_date in result  # 取得日時

    def test_market_cap_none_shows_na(self):
        info = StockInfo(
            code="9999", ticker="9999.T", name="テスト社",
            current_price=1000.0, previous_close=1000.0,
            open_price=1000.0, day_high=1010.0, day_low=990.0,
            volume=10000, market_cap=None,
        )
        result = _build_info_text(_make_ohlcv(), info)
        assert "N/A" in result

    def test_negative_change_shows_minus_sign(self):
        info = StockInfo(
            code="9999", ticker="9999.T", name="テスト社",
            current_price=990.0, previous_close=1000.0,
            open_price=1000.0, day_high=1005.0, day_low=985.0,
            volume=10000, market_cap=None,
        )
        result = _build_info_text(_make_ohlcv(), info)
        assert "-10" in result


# ---------------------------------------------------------------------------
# plot_intraday_live() のテスト
# ---------------------------------------------------------------------------

def _make_empty_ohlcv() -> pd.DataFrame:
    """空の OHLCV DataFrame を作る。"""
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


class TestPlotIntradayLive:
    """plot_intraday_live() が FuncAnimation でチャートを自動更新するケース"""

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_funcanimation_is_created_and_plt_show_is_called_with_block(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        assert mock_anim.called
        mock_show.assert_called_once_with(block=True)

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_funcanimation_interval_is_interval_sec_times_1000(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=30)
        _, kwargs = mock_anim.call_args
        assert kwargs.get("interval") == 30 * 1000

    @patch("src.chart.fetch_intraday_data")
    def test_raises_value_error_on_empty_initial_data(self, mock_fetch):
        mock_fetch.return_value = _make_empty_ohlcv()
        with pytest.raises(ValueError):
            plot_intraday_live("7203", info=_make_stock_info())

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_skips_when_data_is_empty(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        # 1回目（初期取得）は正常データ、2回目（animate内）は空
        mock_fetch.side_effect = [_make_ohlcv(), _make_empty_ohlcv()]
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        # 空データなので ax1.clear() は呼ばれない
        assert not mock_axes[0].clear.called

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_clears_axes_on_new_data(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        assert mock_axes[0].clear.called
        assert mock_axes[2].clear.called

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_updates_info_text(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_fig = MagicMock()
        name_mock   = MagicMock(name="name_txt")
        price_mock  = MagicMock(name="price_txt")
        change_mock = MagicMock(name="change_txt")
        meta_mock   = MagicMock(name="meta_txt")
        mock_fig.text.side_effect = [name_mock, price_mock, change_mock, meta_mock]
        mock_plot.return_value = (mock_fig, mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        # price / change / meta の各テキストオブジェクトが更新されること
        assert price_mock.set_text.called
        assert change_mock.set_text.called
        assert change_mock.set_color.called
        assert meta_mock.set_text.called
        # price は info.current_price（3410.0）を反映した値になること
        price_call_arg = price_mock.set_text.call_args[0][0]
        assert "3,410" in price_call_arg

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_live_label_in_row1_text(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        all_texts = " ".join(c[0][2] for c in mock_fig.text.call_args_list)
        assert "自動更新" in all_texts

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_initial_frame_calls_detect_pattern_when_predict_true(
        self, mock_fetch, mock_plot, mock_anim, mock_show, mock_detect, mock_draw
    ):
        """predict=True のとき初期フレーム描画時に detect_pattern が呼ばれること"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        mock_detect.return_value = None
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60, predict=True)
        assert mock_detect.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_initial_frame_calls_draw_prediction_when_pattern_found(
        self, mock_fetch, mock_plot, mock_anim, mock_show, mock_detect, mock_draw
    ):
        """predict=True かつパターン検出時、初期フレームで draw_prediction が呼ばれること"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        mock_detect.return_value = MagicMock()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60, predict=True)
        assert mock_draw.called

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_calls_draw_idle(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        """animate() 呼び出し後に fig.canvas.draw_idle が呼ばれること"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        mock_fig.canvas.draw_idle.assert_called()

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_uses_stock_info_price_not_ohlcv_close(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        """animate() 後の price テキストは OHLCV の Close[-1] ではなく
        info.current_price（fetch_stock_info 由来）を使うこと"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_fig = MagicMock()
        name_mock   = MagicMock(name="name_txt")
        price_mock  = MagicMock(name="price_txt")
        change_mock = MagicMock(name="change_txt")
        meta_mock   = MagicMock(name="meta_txt")
        mock_fig.text.side_effect = [name_mock, price_mock, change_mock, meta_mock]
        mock_plot.return_value = (mock_fig, mock_axes)
        # _make_ohlcv() の Close[-1] = 3412.0、_make_stock_info() の current_price = 3410.0
        # → price テキストは 3410.0 (info 由来) になること
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        price_call_arg = price_mock.set_text.call_args[0][0]
        assert "3,410" in price_call_arg
        assert "3,412" not in price_call_arg


# ---------------------------------------------------------------------------
# _build_info_parts() のテスト
# ---------------------------------------------------------------------------

class TestBuildInfoParts:
    """_build_info_parts() が正しい情報パーツを生成するケース"""

    def test_name_contains_company_name_and_ticker(self):
        result = _build_info_parts(_make_ohlcv(), _make_stock_info())
        assert "トヨタ自動車" in result["name"]
        assert "7203.T" in result["name"]

    def test_price_contains_yen_and_value(self):
        result = _build_info_parts(_make_ohlcv(), _make_stock_info())
        assert "¥" in result["price"]
        assert "3,410" in result["price"]

    def test_change_color_is_green_when_positive(self):
        # _make_stock_info() は current_price=3410, previous_close=3392 → diff=+18
        result = _build_info_parts(_make_ohlcv(), _make_stock_info())
        assert result["change_color"] == "#22ab94"

    def test_change_color_is_red_when_negative(self):
        info = StockInfo(
            code="9999", ticker="9999.T", name="テスト社",
            current_price=990.0, previous_close=1000.0,
            open_price=1000.0, day_high=1005.0, day_low=985.0,
            volume=10000, market_cap=None,
        )
        result = _build_info_parts(_make_ohlcv(), info)
        assert result["change_color"] == "#f7525f"

    def test_change_color_is_green_when_zero(self):
        info = StockInfo(
            code="9999", ticker="9999.T", name="テスト社",
            current_price=1000.0, previous_close=1000.0,
            open_price=1000.0, day_high=1005.0, day_low=995.0,
            volume=10000, market_cap=None,
        )
        result = _build_info_parts(_make_ohlcv(), info)
        assert result["change_color"] == "#22ab94"

    def test_meta_contains_volume_and_timestamp(self):
        df = _make_ohlcv()
        result = _build_info_parts(df, _make_stock_info())
        assert "560,000" in result["meta"]
        expected_date = df.index[-1].strftime("%Y-%m-%d")
        assert expected_date in result["meta"]

    def test_meta_contains_na_when_market_cap_is_none(self):
        info = StockInfo(
            code="9999", ticker="9999.T", name="テスト社",
            current_price=1000.0, previous_close=1000.0,
            open_price=1000.0, day_high=1005.0, day_low=995.0,
            volume=10000, market_cap=None,
        )
        result = _build_info_parts(_make_ohlcv(), info)
        assert "N/A" in result["meta"]


# ---------------------------------------------------------------------------
# plot_intraday() の predict フラグ検証
# ---------------------------------------------------------------------------

class TestPlotIntradayPredict:
    """plot_intraday() が predict=True/False を正しく処理するケース"""

    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_false_does_not_call_detect_pattern(self, mock_plot, mock_show):
        """predict が指定されない（デフォルト False）場合、正常完了すること"""
        mock_plot.return_value = (MagicMock(), [MagicMock()] * 4)
        plot_intraday(_make_ohlcv(), info=_make_stock_info())
        assert mock_show.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_true_calls_canvas_draw_idle_before_show(
        self, mock_plot, mock_show, mock_detect, mock_draw
    ):
        """predict=True のとき mpf.show() の前に fig.canvas.draw_idle() が呼ばれること"""
        mock_fig = MagicMock()
        mock_plot.return_value = (mock_fig, [MagicMock()] * 4)
        mock_detect.return_value = None
        call_order = []
        mock_fig.canvas.draw_idle.side_effect = lambda: call_order.append("draw_idle")
        mock_show.side_effect = lambda: call_order.append("show")
        plot_intraday(_make_ohlcv(), info=_make_stock_info(), predict=True)
        assert "draw_idle" in call_order
        assert call_order.index("draw_idle") < call_order.index("show")

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_true_calls_detect_pattern(
        self, mock_plot, mock_show, mock_detect, mock_draw
    ):
        """predict=True のとき detect_pattern が呼ばれること"""
        mock_plot.return_value = (MagicMock(), [MagicMock()] * 4)
        mock_detect.return_value = None
        plot_intraday(_make_ohlcv(), info=_make_stock_info(), predict=True)
        assert mock_detect.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_true_calls_draw_prediction_when_pattern_found(
        self, mock_plot, mock_show, mock_detect, mock_draw
    ):
        """パターン検出あり → draw_prediction が呼ばれること"""
        mock_plot.return_value = (MagicMock(), [MagicMock()] * 4)
        mock_detect.return_value = MagicMock()  # non-None = pattern found
        plot_intraday(_make_ohlcv(), info=_make_stock_info(), predict=True)
        assert mock_draw.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_true_no_draw_when_no_pattern(
        self, mock_plot, mock_show, mock_detect, mock_draw
    ):
        """パターン未検出 → draw_prediction が呼ばれないこと"""
        mock_plot.return_value = (MagicMock(), [MagicMock()] * 4)
        mock_detect.return_value = None
        plot_intraday(_make_ohlcv(), info=_make_stock_info(), predict=True)
        assert not mock_draw.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.mpf.show")
    @patch("src.chart.mpf.plot")
    def test_predict_true_no_pattern_shows_in_chart(
        self, mock_plot, mock_show, mock_detect, mock_draw
    ):
        """パターン未検出 → axes[0].text に「パターン未検出」が表示されること"""
        mock_axes = [MagicMock() for _ in range(4)]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_detect.return_value = None
        plot_intraday(_make_ohlcv(), info=_make_stock_info(), predict=True)
        assert mock_axes[0].text.called
        text_arg = mock_axes[0].text.call_args[0][2]
        assert "パターン未検出" in text_arg


# ---------------------------------------------------------------------------
# plot_intraday_live() の predict フラグ検証
# ---------------------------------------------------------------------------

class TestPlotIntradayLivePredict:
    """plot_intraday_live() が predict=True/False を正しく animate() 内で処理するケース"""

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_calls_detect_pattern_when_predict_true(
        self, mock_fetch, mock_plot, mock_anim, mock_show, mock_detect, mock_draw
    ):
        """predict=True → animate() 内で detect_pattern が呼ばれること"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        mock_detect.return_value = None
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60, predict=True)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        assert mock_detect.called

    @patch("src.predictor.draw_prediction")
    @patch("src.predictor.detect_pattern")
    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_calls_draw_prediction_when_pattern_found(
        self, mock_fetch, mock_plot, mock_anim, mock_show, mock_detect, mock_draw
    ):
        """パターン検出あり → animate() 内で draw_prediction が呼ばれること"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        mock_detect.return_value = MagicMock()  # pattern found
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60, predict=True)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        assert mock_draw.called

    @patch("src.chart.plt.show")
    @patch("src.chart.animation.FuncAnimation")
    @patch("src.chart.mpf.plot")
    @patch("src.chart.fetch_intraday_data")
    def test_animate_does_not_call_detect_pattern_when_predict_false(
        self, mock_fetch, mock_plot, mock_anim, mock_show
    ):
        """predict=False → animate() 内で src.predictor をインポートしないこと"""
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_plot.return_value = (MagicMock(), mock_axes)
        mock_fetch.return_value = _make_ohlcv()
        plot_intraday_live("7203", info=_make_stock_info(), interval_sec=60)
        animate_fn = mock_anim.call_args[0][1]
        animate_fn(0)
        assert mock_show.called  # 正常完了すること
