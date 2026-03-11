from src.models import StockInfo


def _format_change(current: float, previous: float) -> str:
    diff = current - previous
    pct = (diff / previous * 100) if previous else 0.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:,.0f} ({sign}{pct:.2f}%)"


def _format_market_cap(market_cap: int | None) -> str:
    if market_cap is None:
        return "N/A"
    if market_cap >= 1_000_000_000_000:
        return f"¥{market_cap / 1_000_000_000_000:.1f}兆"
    if market_cap >= 100_000_000:
        return f"¥{market_cap / 100_000_000:.0f}億"
    return f"¥{market_cap:,}"


def format_stock_info(info: StockInfo) -> str:
    change_str = _format_change(info.current_price, info.previous_close)
    cap_str = _format_market_cap(info.market_cap)

    lines = [
        f"銘柄名      : {info.name} ({info.ticker})",
        f"現在値      : ¥{info.current_price:,.0f}",
        f"前日比      : {change_str}",
        f"始値/高値/安値: ¥{info.open_price:,.0f} / ¥{info.day_high:,.0f} / ¥{info.day_low:,.0f}",
        f"出来高      : {info.volume:,}",
        f"時価総額    : {cap_str}",
    ]
    return "\n".join(lines)


def print_stock_info(info: StockInfo) -> None:
    print(format_stock_info(info))
