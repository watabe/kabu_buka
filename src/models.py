from dataclasses import dataclass


@dataclass
class StockInfo:
    code: str             # 入力銘柄コード (例: "7203")
    ticker: str           # yfinance ティッカー (例: "7203.T")
    name: str             # 銘柄名
    current_price: float
    previous_close: float
    open_price: float
    day_high: float
    day_low: float
    volume: int
    market_cap: int | None
