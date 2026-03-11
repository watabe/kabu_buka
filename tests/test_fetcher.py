"""
Tests for src/fetcher.py

Tests use unittest.mock to avoid real network calls.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.models import StockInfo
from src.fetcher import fetch_stock_info, fetch_intraday_data, StockNotFoundError, FetchError


MOCK_INFO_TOYOTA = {
    "longName": "Toyota Motor Corporation",
    "currentPrice": 3420.0,
    "previousClose": 3390.0,
    "open": 3400.0,
    "dayHigh": 3435.0,
    "dayLow": 3398.0,
    "volume": 8234500,
    "marketCap": 55_600_000_000_000,
}


def _make_ticker_mock(info_dict: dict) -> MagicMock:
    ticker = MagicMock()
    ticker.info = info_dict
    return ticker


class TestFetchStockInfoSuccess:
    """fetch_stock_info() が正常な StockInfo を返すケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_returns_stock_info_dataclass(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_ticker_mock(MOCK_INFO_TOYOTA)

        result = fetch_stock_info("7203")

        assert isinstance(result, StockInfo)

    @patch("src.fetcher.yf.Ticker")
    def test_ticker_suffix_t_is_appended(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_ticker_mock(MOCK_INFO_TOYOTA)

        fetch_stock_info("7203")

        mock_ticker_cls.assert_called_once_with("7203.T")

    @patch("src.fetcher.yf.Ticker")
    def test_stock_info_fields_are_populated(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_ticker_mock(MOCK_INFO_TOYOTA)

        result = fetch_stock_info("7203")

        assert result.code == "7203"
        assert result.ticker == "7203.T"
        assert result.name == "Toyota Motor Corporation"
        assert result.current_price == 3420.0
        assert result.previous_close == 3390.0
        assert result.open_price == 3400.0
        assert result.day_high == 3435.0
        assert result.day_low == 3398.0
        assert result.volume == 8234500
        assert result.market_cap == 55_600_000_000_000

    @patch("src.fetcher.yf.Ticker")
    def test_market_cap_can_be_none(self, mock_ticker_cls):
        info = {**MOCK_INFO_TOYOTA, "marketCap": None}
        mock_ticker_cls.return_value = _make_ticker_mock(info)

        result = fetch_stock_info("7203")

        assert result.market_cap is None

    @patch("src.fetcher.yf.Ticker")
    def test_market_cap_missing_key_treated_as_none(self, mock_ticker_cls):
        info = {k: v for k, v in MOCK_INFO_TOYOTA.items() if k != "marketCap"}
        mock_ticker_cls.return_value = _make_ticker_mock(info)

        result = fetch_stock_info("7203")

        assert result.market_cap is None


class TestFetchStockInfoNotFound:
    """存在しない銘柄コードのとき StockNotFoundError が発生するケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_raises_stock_not_found_when_info_empty(self, mock_ticker_cls):
        # yfinance は存在しない銘柄のとき info が空辞書を返す
        mock_ticker_cls.return_value = _make_ticker_mock({})

        with pytest.raises(StockNotFoundError):
            fetch_stock_info("0000")

    @patch("src.fetcher.yf.Ticker")
    def test_raises_stock_not_found_when_long_name_missing(self, mock_ticker_cls):
        # longName がないケースも「銘柄なし」とみなす
        info = {k: v for k, v in MOCK_INFO_TOYOTA.items() if k != "longName"}
        mock_ticker_cls.return_value = _make_ticker_mock(info)

        with pytest.raises(StockNotFoundError):
            fetch_stock_info("0000")

    @patch("src.fetcher.yf.Ticker")
    def test_error_message_contains_code(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_ticker_mock({})

        with pytest.raises(StockNotFoundError, match="0000"):
            fetch_stock_info("0000")


class TestFetchStockInfoNetworkError:
    """ネットワークエラー・タイムアウト時に FetchError が発生するケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_raises_fetch_error_on_connection_error(self, mock_ticker_cls):
        import requests
        ticker = MagicMock()
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("connection refused")
        ))
        mock_ticker_cls.return_value = ticker

        with pytest.raises(FetchError):
            fetch_stock_info("7203")

    @patch("src.fetcher.yf.Ticker")
    def test_raises_fetch_error_on_timeout(self, mock_ticker_cls):
        import requests
        ticker = MagicMock()
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(
            requests.exceptions.Timeout("timed out")
        ))
        mock_ticker_cls.return_value = ticker

        with pytest.raises(FetchError):
            fetch_stock_info("7203")


# ---------------------------------------------------------------------------
# fetch_intraday_data() のテスト
# ---------------------------------------------------------------------------

def _make_history_df(n: int = 3) -> pd.DataFrame:
    """yfinance が返すような OHLCV DataFrame を作る。"""
    index = pd.date_range(
        "2026-02-20 09:00", periods=n, freq="1min", tz="Asia/Tokyo"
    )
    return pd.DataFrame(
        {
            "Open":         [3400.0, 3410.0, 3405.0][:n],
            "High":         [3415.0, 3420.0, 3418.0][:n],
            "Low":          [3395.0, 3405.0, 3400.0][:n],
            "Close":        [3410.0, 3405.0, 3412.0][:n],
            "Volume":       [100000, 120000,  95000][:n],
            "Dividends":    [0.0, 0.0, 0.0][:n],
            "Stock Splits": [0.0, 0.0, 0.0][:n],
        },
        index=index,
    )


class TestFetchIntradayDataSuccess:
    """fetch_intraday_data() が正常な DataFrame を返すケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_returns_dataframe(self, mock_ticker_cls):
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(return_value=_make_history_df())
        )

        result = fetch_intraday_data("7203")

        assert isinstance(result, pd.DataFrame)

    @patch("src.fetcher.yf.Ticker")
    def test_ticker_suffix_t_is_appended(self, mock_ticker_cls):
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(return_value=_make_history_df())
        )

        fetch_intraday_data("7203")

        mock_ticker_cls.assert_called_once_with("7203.T")

    @patch("src.fetcher.yf.Ticker")
    def test_history_called_with_correct_params(self, mock_ticker_cls):
        mock_history = MagicMock(return_value=_make_history_df())
        mock_ticker_cls.return_value = MagicMock(history=mock_history)

        fetch_intraday_data("7203")

        mock_history.assert_called_once_with(period="1d", interval="1m")

    @patch("src.fetcher.yf.Ticker")
    def test_result_has_required_columns(self, mock_ticker_cls):
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(return_value=_make_history_df())
        )

        result = fetch_intraday_data("7203")

        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    @patch("src.fetcher.yf.Ticker")
    def test_result_index_is_datetime(self, mock_ticker_cls):
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(return_value=_make_history_df())
        )

        result = fetch_intraday_data("7203")

        assert isinstance(result.index, pd.DatetimeIndex)


class TestFetchIntradayDataEmpty:
    """取引時間外・休日で history が空 DataFrame を返すケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_returns_empty_dataframe_when_no_data(self, mock_ticker_cls):
        empty = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume",
                     "Dividends", "Stock Splits"]
        )
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(return_value=empty)
        )

        result = fetch_intraday_data("7203")

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestFetchIntradayDataNetworkError:
    """ネットワークエラー時に FetchError が発生するケース"""

    @patch("src.fetcher.yf.Ticker")
    def test_raises_fetch_error_on_connection_error(self, mock_ticker_cls):
        import requests
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(
                side_effect=requests.exceptions.ConnectionError("fail")
            )
        )

        with pytest.raises(FetchError):
            fetch_intraday_data("7203")

    @patch("src.fetcher.yf.Ticker")
    def test_raises_fetch_error_on_timeout(self, mock_ticker_cls):
        import requests
        mock_ticker_cls.return_value = MagicMock(
            history=MagicMock(
                side_effect=requests.exceptions.Timeout("timed out")
            )
        )

        with pytest.raises(FetchError):
            fetch_intraday_data("7203")
