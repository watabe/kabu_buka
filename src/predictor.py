"""チャートパターン検出と将来予測ライン生成モジュール。

scipy への依存なし。numpy のスライディングウィンドウで極値を検出する。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PatternResult:
    name:            str               # 例: "ダブルトップ"
    direction:       str               # "up" または "down"
    forecast_prices: list             # 予測価格列（10本）
    forecast_index:  pd.DatetimeIndex  # 予測時刻列（将来タイムスタンプ）


def _local_maxima(series: np.ndarray, order: int = 5) -> np.ndarray:
    """スライディングウィンドウで極大インデックスを返す。

    点 i が order 本の前後すべての点より厳密に大きい場合に極大とする。
    """
    n = len(series)
    result = []
    for i in range(order, n - order):
        neighborhood = np.concatenate([series[i - order:i], series[i + 1:i + order + 1]])
        if series[i] > neighborhood.max():
            result.append(i)
    return np.array(result, dtype=int)


def _local_minima(series: np.ndarray, order: int = 5) -> np.ndarray:
    """スライディングウィンドウで極小インデックスを返す。

    点 i が order 本の前後すべての点より厳密に小さい場合に極小とする。
    """
    n = len(series)
    result = []
    for i in range(order, n - order):
        neighborhood = np.concatenate([series[i - order:i], series[i + 1:i + order + 1]])
        if series[i] < neighborhood.min():
            result.append(i)
    return np.array(result, dtype=int)


def _extrapolate_index(df: pd.DataFrame, n_steps: int = 10) -> pd.DatetimeIndex:
    """最終タイムスタンプから 1 分足で n_steps 本先の DatetimeIndex を生成する。"""
    last_ts = df.index[-1]
    freq = pd.tseries.frequencies.to_offset("1min")
    return pd.date_range(start=last_ts + freq, periods=n_steps, freq=freq, tz=last_ts.tz)


def _make_forecast(last_close: float, direction: str, n: int = 10) -> list:
    """direction に応じて 0.5%/本 の線形外挿で予測価格列を生成する。"""
    if direction == "up":
        return [last_close * (1.0 + 0.005 * (i + 1)) for i in range(n)]
    else:
        return [last_close * (1.0 - 0.005 * (i + 1)) for i in range(n)]


def detect_pattern(df: pd.DataFrame) -> Optional[PatternResult]:
    """Close 系列からチャートパターンを検出する。

    データが 30 本未満の場合は None を返す。
    検出優先度: ヘッドアンドショルダー > 逆ヘッドアンドショルダー
                > ダブルトップ > ダブルボトム

    Args:
        df: Open/High/Low/Close/Volume 列と DatetimeIndex を持つ DataFrame

    Returns:
        PatternResult（パターン検出時）または None（未検出またはデータ不足）
    """
    if len(df) < 30:
        return None

    close = df["Close"].values.astype(float)
    maxima = _local_maxima(close)
    minima = _local_minima(close)
    last_close = float(close[-1])
    forecast_index = _extrapolate_index(df)
    n_steps = len(forecast_index)

    # 1. ヘッドアンドショルダー: 極大 3 つ以上、中央が最高、両肩の価格差 ≤ 3%
    if len(maxima) >= 3:
        l_idx, h_idx, r_idx = maxima[-3], maxima[-2], maxima[-1]
        lv, hv, rv = close[l_idx], close[h_idx], close[r_idx]
        if hv > lv and hv > rv:
            shoulder_diff = abs(lv - rv) / max(lv, rv)
            if shoulder_diff <= 0.03:
                return PatternResult(
                    name="ヘッドアンドショルダー",
                    direction="down",
                    forecast_prices=_make_forecast(last_close, "down", n_steps),
                    forecast_index=forecast_index,
                )

    # 2. 逆ヘッドアンドショルダー: 極小 3 つ以上、中央が最低、両肩の価格差 ≤ 3%
    if len(minima) >= 3:
        l_idx, h_idx, r_idx = minima[-3], minima[-2], minima[-1]
        lv, hv, rv = close[l_idx], close[h_idx], close[r_idx]
        if hv < lv and hv < rv:
            shoulder_diff = abs(lv - rv) / max(lv, rv)
            if shoulder_diff <= 0.03:
                return PatternResult(
                    name="逆ヘッドアンドショルダー",
                    direction="up",
                    forecast_prices=_make_forecast(last_close, "up", n_steps),
                    forecast_index=forecast_index,
                )

    # 3. ダブルトップ: 極大 2 つ以上、価格差 ≤ 2%、間に極小が存在
    if len(maxima) >= 2:
        p1, p2 = maxima[-2], maxima[-1]
        v1, v2 = close[p1], close[p2]
        peak_diff = abs(v1 - v2) / max(v1, v2)
        if peak_diff <= 0.02:
            between_minima = minima[(minima > p1) & (minima < p2)]
            if len(between_minima) > 0:
                return PatternResult(
                    name="ダブルトップ",
                    direction="down",
                    forecast_prices=_make_forecast(last_close, "down", n_steps),
                    forecast_index=forecast_index,
                )

    # 4. ダブルボトム: 極小 2 つ以上、価格差 ≤ 2%、間に極大が存在
    if len(minima) >= 2:
        t1, t2 = minima[-2], minima[-1]
        v1, v2 = close[t1], close[t2]
        trough_diff = abs(v1 - v2) / max(v1, v2)
        if trough_diff <= 0.02:
            between_maxima = maxima[(maxima > t1) & (maxima < t2)]
            if len(between_maxima) > 0:
                return PatternResult(
                    name="ダブルボトム",
                    direction="up",
                    forecast_prices=_make_forecast(last_close, "up", n_steps),
                    forecast_index=forecast_index,
                )

    return None


def draw_prediction(ax, result: PatternResult, n_existing: int) -> None:
    """予測ラインとパターン名を axes に描画する。

    mplfinance の x 軸は整数インデックス（0, 1, …, n-1）のため、
    将来 10 本の位置を n_existing 起点の整数オフセットで指定する。

    Args:
        ax:         描画対象の matplotlib Axes（ローソク足パネル）
        result:     detect_pattern() が返した PatternResult
        n_existing: 既存データの本数（x 軸の起点インデックス）
    """
    n = len(result.forecast_prices)
    x_values = list(range(n_existing, n_existing + n))
    ax.plot(
        x_values,
        result.forecast_prices,
        linestyle="--",
        alpha=0.7,
        color="#e67e22",
    )
    ax.text(
        0.98, 0.97,
        result.name,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#e67e22",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
