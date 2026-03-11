import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from src.fetcher import fetch_intraday_data
from src.models import StockInfo

_JP_FONT_CANDIDATES = [
    "Hiragino Sans",            # macOS 標準（全バージョン）
    "Hiragino Kaku Gothic Pro",
    "Hiragino Maru Gothic Pro",
    "Arial Unicode MS",         # macOS + MS Office
    "Noto Sans CJK JP",         # 手動インストール / Linux
    "IPAGothic",
    "IPAexGothic",
    "Yu Gothic",                # Windows
    "Meiryo",                   # Windows
]


def _resolve_japanese_font() -> str | None:
    """利用可能な日本語フォントを優先リストから最初に見つけたものを返す。

    Returns:
        フォント名（文字列）、または候補が1つも見つからない場合は None
    """
    available = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in _JP_FONT_CANDIDATES if f in available), None)


def _format_market_cap(market_cap: int | None) -> str:
    """時価総額を人間が読みやすい文字列にフォーマットする。"""
    if market_cap is None:
        return "N/A"
    if market_cap >= 1_000_000_000_000:
        return f"¥{market_cap / 1_000_000_000_000:.1f}兆"
    if market_cap >= 100_000_000:
        return f"¥{market_cap / 100_000_000:.0f}億"
    return f"¥{market_cap:,}"


def _build_info_text(df: pd.DataFrame, info: StockInfo) -> str:
    """チャート下部に表示する情報テキストを組み立てる。

    Args:
        df:   最新の OHLCV DataFrame（最終行のタイムスタンプを取得日時として使用）
        info: 銘柄情報（StockInfo）

    Returns:
        チャート下部に表示する1行のテキスト文字列
    """
    diff = info.current_price - info.previous_close
    pct = (diff / info.previous_close * 100) if info.previous_close else 0.0
    sign = "+" if diff >= 0 else ""
    cap = _format_market_cap(info.market_cap)
    ts = df.index[-1].strftime("%Y-%m-%d %H:%M")
    return (
        f"{info.name} ({info.ticker})  "
        f"現在値: ¥{info.current_price:,.0f}  "
        f"前日比: {sign}{diff:,.0f} ({sign}{pct:.2f}%)  "
        f"出来高: {info.volume:,}  "
        f"時価総額: {cap}  "
        f"取得: {ts}"
    )


def _build_info_parts(df: pd.DataFrame, info: StockInfo) -> dict:
    """チャート上部の各テキストパーツを辞書で返す。

    Returns:
        {
            "name":         str,  # "{info.name} ({info.ticker})"
            "price":        str,  # "¥{current_price:,.0f}"
            "change":       str,  # "{sign}{diff:,.0f} ({sign}{pct:.2f}%)"
            "change_color": str,  # "#22ab94"（diff >= 0）または "#f7525f"（diff < 0）
            "meta":         str,  # "出来高: xxx  時価総額: xxx  取得: YYYY-MM-DD HH:MM"
        }
    """
    diff = info.current_price - info.previous_close
    pct  = (diff / info.previous_close * 100) if info.previous_close else 0.0
    sign = "+" if diff >= 0 else ""
    ts   = df.index[-1].strftime("%Y-%m-%d %H:%M")
    return {
        "name":         f"{info.name} ({info.ticker})",
        "price":        f"¥{info.current_price:,.0f}",
        "change":       f"{sign}{diff:,.0f} ({sign}{pct:.2f}%)",
        "change_color": "#22ab94" if diff >= 0 else "#f7525f",
        "meta": (
            f"出来高: {info.volume:,}  "
            f"時価総額: {_format_market_cap(info.market_cap)}  "
            f"取得: {ts}"
        ),
    }


def plot_intraday(df: pd.DataFrame, info: StockInfo, predict: bool = False) -> None:
    """1分足 OHLCV DataFrame をローソク足チャートでGUI表示する。

    日本語フォントを自動検出してチャートに適用する。フォントが見つからない
    環境でも axes.unicode_minus=False を設定しクラッシュせず動作する。
    チャート下部に銘柄名・株価・取得日時などの会社情報を表示する。

    Args:
        df:      Open/High/Low/Close/Volume 列と DatetimeIndex を持つ DataFrame
        info:    表示する銘柄情報（StockInfo）
        predict: True のときチャートパターンを検出して予測オーバーレイを描画する

    Raises:
        ValueError: df が空の場合
    """
    if df.empty:
        raise ValueError(f"表示するデータがありません: {info.ticker}")

    plot_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    font = _resolve_japanese_font()
    rc: dict = {"axes.unicode_minus": False}
    if font:
        rc["font.family"] = font

    style = mpf.make_mpf_style(base_mpf_style="yahoo", rc=rc)

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        title="",
        ylabel="株価 (円)",
        ylabel_lower="出来高",
        volume=True,
        figsize=(14, 7),
        show_nontrading=False,
        returnfig=True,
    )

    fig.subplots_adjust(top=0.85)
    parts = _build_info_parts(plot_df, info)
    fig.text(0.01, 0.960, parts["name"] + "  1日 (1分足)",  fontsize=13, fontweight="bold", va="top", color="#333333")
    fig.text(0.01, 0.922, parts["price"],                   fontsize=16, fontweight="bold", va="top", color="#333333")
    fig.text(0.20, 0.922, parts["change"],                  fontsize=12, va="top", color=parts["change_color"])
    fig.text(0.01, 0.885, parts["meta"],   fontsize=9,  va="top", color="#666666")

    if predict:
        from src.predictor import detect_pattern, draw_prediction
        result = detect_pattern(plot_df)
        if result:
            draw_prediction(axes[0], result, n_existing=len(plot_df))
        else:
            axes[0].text(
                0.98, 0.97,
                "パターン未検出",
                transform=axes[0].transAxes,
                ha="right", va="top",
                fontsize=10, color="#888888",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    fig.canvas.draw_idle()
    mpf.show()


def plot_intraday_live(code: str, info: StockInfo, interval_sec: int = 60, predict: bool = False) -> None:
    """指定インターバルごとにチャートを自動更新して表示する。

    ウィンドウを閉じるか Ctrl+C するまでループし続ける。
    取得データが空のフレームはスキップして前回の描画を維持する。

    Args:
        code:         東証銘柄コード (例: "7203")
        info:         初期表示用の銘柄情報（StockInfo）
        interval_sec: 更新インターバル（秒）。デフォルト 60 秒
        predict:      True のとき更新のたびにパターン検出と予測オーバーレイを再描画する

    Raises:
        ValueError: 初回データが空の場合
    """
    df_init = fetch_intraday_data(code)
    if df_init.empty:
        raise ValueError(f"表示するデータがありません: {info.ticker}")

    font = _resolve_japanese_font()
    rc: dict = {"axes.unicode_minus": False}
    if font:
        rc["font.family"] = font
    style = mpf.make_mpf_style(base_mpf_style="yahoo", rc=rc)

    plot_df = df_init[["Open", "High", "Low", "Close", "Volume"]].copy()
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        title="",
        ylabel="株価 (円)",
        ylabel_lower="出来高",
        volume=True,
        figsize=(14, 7),
        show_nontrading=False,
        returnfig=True,
    )
    ax1 = axes[0]
    ax2 = axes[2]

    fig.subplots_adjust(top=0.85)
    parts = _build_info_parts(plot_df, info)
    fig.text(0.01, 0.960, parts["name"] + "  1日 (1分足)  自動更新",
             fontsize=13, fontweight="bold", va="top", color="#333333")
    _price_txt  = fig.text(0.01, 0.922, parts["price"],  fontsize=16, fontweight="bold", va="top", color="#333333")
    _change_txt = fig.text(0.20, 0.922, parts["change"], fontsize=12, va="top", color=parts["change_color"])
    _meta_txt   = fig.text(0.01, 0.885, parts["meta"],   fontsize=9,  va="top", color="#666666")

    if predict:
        from src.predictor import detect_pattern, draw_prediction
        result = detect_pattern(plot_df)
        if result:
            draw_prediction(ax1, result, n_existing=len(plot_df))
        else:
            ax1.text(
                0.98, 0.97,
                "パターン未検出",
                transform=ax1.transAxes,
                ha="right", va="top",
                fontsize=10, color="#888888",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
    fig.canvas.draw_idle()

    def animate(_ival: int) -> None:
        df_new = fetch_intraday_data(code)
        if df_new.empty:
            return
        plot_df_new = df_new[["Open", "High", "Low", "Close", "Volume"]].copy()
        ax1.clear()
        ax2.clear()
        mpf.plot(plot_df_new, ax=ax1, volume=ax2, type="candle", style=style)
        new_parts = _build_info_parts(plot_df_new, info)
        _price_txt.set_text(new_parts["price"])
        _change_txt.set_text(new_parts["change"])
        _change_txt.set_color(new_parts["change_color"])
        _meta_txt.set_text(new_parts["meta"])
        if predict:
            from src.predictor import detect_pattern, draw_prediction
            result = detect_pattern(plot_df_new)
            if result:
                draw_prediction(ax1, result, n_existing=len(plot_df_new))
            else:
                ax1.text(
                    0.98, 0.97,
                    "パターン未検出",
                    transform=ax1.transAxes,
                    ha="right", va="top",
                    fontsize=10, color="#888888",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )
        fig.canvas.draw_idle()

    ani = animation.FuncAnimation(  # noqa: F841  (ani はスコープ保持が必要)
        fig, animate,
        interval=interval_sec * 1000,
        cache_frame_data=False,
    )
    plt.show(block=True)
