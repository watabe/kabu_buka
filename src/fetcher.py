import pandas as pd
import requests
import yfinance as yf

from src.models import StockInfo


class StockNotFoundError(Exception):
    pass


class FetchError(Exception):
    pass


def fetch_stock_info(code: str) -> StockInfo:
    """銘柄コードを受け取り yfinance 経由で情報を取得して StockInfo を返す。

    Args:
        code: 東証銘柄コード (例: "7203")

    Raises:
        StockNotFoundError: 銘柄が見つからない場合
        FetchError: ネットワークエラー・タイムアウトの場合
    """
    ticker_symbol = f"{code}.T"
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise FetchError("データ取得に失敗しました。通信状況を確認してください。") from e
    except Exception as e:
        raise FetchError("データ取得に失敗しました。通信状況を確認してください。") from e

    if not info or not info.get("longName"):
        raise StockNotFoundError(f"銘柄が見つかりませんでした: {code}")

    return StockInfo(
        code=code,
        ticker=ticker_symbol,
        name=info["longName"],
        current_price=float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0),
        previous_close=float(info.get("previousClose") or info.get("regularMarketPreviousClose") or 0.0),
        open_price=float(info.get("open") or info.get("regularMarketOpen") or 0.0),
        day_high=float(info.get("dayHigh") or info.get("regularMarketDayHigh") or 0.0),
        day_low=float(info.get("dayLow") or info.get("regularMarketDayLow") or 0.0),
        volume=int(info.get("volume") or info.get("regularMarketVolume") or 0),
        market_cap=info.get("marketCap"),
    )


def fetch_intraday_data(code: str) -> pd.DataFrame:
    """当日1分足 OHLCV データを取得する。

    Returns:
        DataFrame: columns に Open/High/Low/Close/Volume を含む。
                   index は DatetimeIndex (Asia/Tokyo)。
                   取引時間外・休日でデータなしの場合は空 DataFrame を返す。

    Raises:
        FetchError: ネットワークエラー・タイムアウトの場合
    """
    ticker_symbol = f"{code}.T"
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1d", interval="1m")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise FetchError("データ取得に失敗しました。通信状況を確認してください。") from e
    except Exception as e:
        raise FetchError("データ取得に失敗しました。通信状況を確認してください。") from e

    return df
