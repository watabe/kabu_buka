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
    confidence:      float            # 信頼度スコア（0.0〜1.0）


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


def _make_target_forecast(last_close: float, target_price: float, n: int = 10) -> list:
    """last_close から target_price まで n ステップで線形補間した予測価格列を返す。"""
    return list(np.linspace(last_close, target_price, n + 1)[1:])


def _fit_trendline(indices: np.ndarray, values: np.ndarray) -> tuple:
    """1次線形回帰で傾き (slope) と切片 (intercept) を返す。"""
    slope, intercept = np.polyfit(indices.astype(float), values.astype(float), 1)
    return float(slope), float(intercept)


def _pct_range(values: np.ndarray) -> float:
    """(max - min) / min を返す。values が空または min==0 の場合は 0.0。"""
    if len(values) == 0:
        return 0.0
    mn = float(values.min())
    if mn == 0:
        return 0.0
    return float((float(values.max()) - mn) / mn)


def _clamp(value: float) -> float:
    """0.0〜1.0 にクランプする。"""
    return max(0.0, min(1.0, value))


def detect_pattern(df: pd.DataFrame) -> Optional[PatternResult]:
    """Close 系列からチャートパターンを検出する。

    データが 30 本未満の場合は None を返す。
    検出優先度:
        ヘッドアンドショルダー > 逆ヘッドアンドショルダー
        > 上昇三角形 > 下降三角形 > 対称三角形
        > ライジングウェッジ > フォーリングウェッジ
        > トリプルトップ > トリプルボトム
        > 強気フラッグ > 弱気フラッグ
        > カップアンドハンドル
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
                neck_cands = minima[(minima > l_idx) & (minima < r_idx)]
                if len(neck_cands) > 0:
                    neckline = float(close[neck_cands].mean())
                else:
                    neckline = float(min(lv, rv))
                target     = neckline - (hv - neckline)
                confidence = _clamp(1.0 - shoulder_diff / 0.03)
                return PatternResult(
                    name="ヘッドアンドショルダー",
                    direction="down",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 2. 逆ヘッドアンドショルダー: 極小 3 つ以上、中央が最低、両肩の価格差 ≤ 3%
    if len(minima) >= 3:
        l_idx, h_idx, r_idx = minima[-3], minima[-2], minima[-1]
        lv, hv, rv = close[l_idx], close[h_idx], close[r_idx]
        if hv < lv and hv < rv:
            shoulder_diff = abs(lv - rv) / max(lv, rv)
            if shoulder_diff <= 0.03:
                neck_cands = maxima[(maxima > l_idx) & (maxima < r_idx)]
                if len(neck_cands) > 0:
                    neckline = float(close[neck_cands].mean())
                else:
                    neckline = float(max(lv, rv))
                target     = neckline + (neckline - hv)
                confidence = _clamp(1.0 - shoulder_diff / 0.03)
                return PatternResult(
                    name="逆ヘッドアンドショルダー",
                    direction="up",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 三角形・ウェッジ系: 極大・極小それぞれ 3 点以上が必要
    # flat_thresh = 0.1%/bar (price-normalized)
    flat_thresh = float(close.mean()) * 0.001
    if len(maxima) >= 3 and len(minima) >= 3:
        slope_high, _ = _fit_trendline(maxima[-3:], close[maxima[-3:]])
        slope_low, _  = _fit_trendline(minima[-3:], close[minima[-3:]])
        max_width = abs(float(close[maxima[-3]]) - float(close[minima[-3]]))

        # 3. 上昇三角形: 抵抗線が水平 + 支持線が上昇
        if abs(slope_high) <= flat_thresh and slope_low > flat_thresh:
            target     = last_close + max_width
            confidence = _clamp(1.0 - abs(slope_high) / flat_thresh)
            return PatternResult(
                name="上昇三角形",
                direction="up",
                forecast_prices=_make_target_forecast(last_close, target, n_steps),
                forecast_index=forecast_index,
                confidence=confidence,
            )

        # 4. 下降三角形: 抵抗線が下降 + 支持線が水平
        if slope_high < -flat_thresh and abs(slope_low) <= flat_thresh:
            target     = last_close - max_width
            confidence = _clamp(1.0 - abs(slope_low) / flat_thresh)
            return PatternResult(
                name="下降三角形",
                direction="down",
                forecast_prices=_make_target_forecast(last_close, target, n_steps),
                forecast_index=forecast_index,
                confidence=confidence,
            )

        # 5. 対称三角形: 抵抗線が下降 + 支持線が上昇（収束）
        if slope_high < -flat_thresh and slope_low > flat_thresh:
            look_back = min(20, len(close))
            recent_slope, _ = np.polyfit(
                np.arange(look_back, dtype=float),
                close[-look_back:],
                1,
            )
            direction  = "up" if recent_slope >= 0 else "down"
            target     = last_close + max_width if direction == "up" else last_close - max_width
            abs_high   = abs(slope_high)
            abs_low    = abs(slope_low)
            confidence = _clamp(min(abs_high, abs_low) / max(abs_high, abs_low))
            return PatternResult(
                name="対称三角形",
                direction=direction,
                forecast_prices=_make_target_forecast(last_close, target, n_steps),
                forecast_index=forecast_index,
                confidence=confidence,
            )

        # 6. ライジングウェッジ: 両方上昇、谷の傾きが峰より急（収束）→ 下落転換
        if slope_high > flat_thresh and slope_low > flat_thresh and slope_low > slope_high:
            target     = last_close - max_width
            confidence = _clamp(1.0 - slope_high / slope_low)
            return PatternResult(
                name="ライジングウェッジ",
                direction="down",
                forecast_prices=_make_target_forecast(last_close, target, n_steps),
                forecast_index=forecast_index,
                confidence=confidence,
            )

        # 7. フォーリングウェッジ: 両方下落、峰の傾きが谷より急（収束）→ 上昇転換
        if slope_high < -flat_thresh and slope_low < -flat_thresh and slope_high < slope_low:
            target     = last_close + max_width
            confidence = _clamp(1.0 - slope_low / slope_high)
            return PatternResult(
                name="フォーリングウェッジ",
                direction="up",
                forecast_prices=_make_target_forecast(last_close, target, n_steps),
                forecast_index=forecast_index,
                confidence=confidence,
            )

    # 8. トリプルトップ: 極大 3 つ以上、すべて価格差 ≤ 3%、間に極小あり、中央が最高でない
    if len(maxima) >= 3:
        p1, p2, p3 = maxima[-3], maxima[-2], maxima[-1]
        v1, v2, v3 = close[p1], close[p2], close[p3]
        diff12 = abs(v1 - v2) / max(v1, v2)
        diff23 = abs(v2 - v3) / max(v2, v3)
        if diff12 <= 0.03 and diff23 <= 0.03 and not (v2 > v1 and v2 > v3):
            b12 = minima[(minima > p1) & (minima < p2)]
            b23 = minima[(minima > p2) & (minima < p3)]
            if len(b12) > 0 and len(b23) > 0:
                neckline   = min(float(close[b12].min()), float(close[b23].min()))
                peak_avg   = (v1 + v2 + v3) / 3.0
                target     = neckline - (peak_avg - neckline)
                confidence = _clamp(1.0 - max(diff12, diff23) / 0.03)
                return PatternResult(
                    name="トリプルトップ",
                    direction="down",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 9. トリプルボトム: 極小 3 つ以上、すべて価格差 ≤ 3%、間に極大あり、中央が最低でない
    if len(minima) >= 3:
        t1, t2, t3 = minima[-3], minima[-2], minima[-1]
        v1, v2, v3 = close[t1], close[t2], close[t3]
        diff12 = abs(v1 - v2) / max(v1, v2)
        diff23 = abs(v2 - v3) / max(v2, v3)
        if diff12 <= 0.03 and diff23 <= 0.03 and not (v2 < v1 and v2 < v3):
            b12 = maxima[(maxima > t1) & (maxima < t2)]
            b23 = maxima[(maxima > t2) & (maxima < t3)]
            if len(b12) > 0 and len(b23) > 0:
                neckline   = max(float(close[b12].max()), float(close[b23].max()))
                trough_avg = (v1 + v2 + v3) / 3.0
                target     = neckline + (neckline - trough_avg)
                confidence = _clamp(1.0 - max(diff12, diff23) / 0.03)
                return PatternResult(
                    name="トリプルボトム",
                    direction="up",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 10. 強気フラッグ / 弱気フラッグ: 急騰・急落ポール + 小幅調整チャンネル
    pole_len = max(len(close) // 3, 10)
    rest = close[pole_len:]
    if len(rest) >= 10:
        pole       = close[:pole_len]
        pole_range = _pct_range(pole)
        rest_range = _pct_range(rest)
        if pole_range >= 0.03 and rest_range < pole_range * 0.5:
            pole_slope, _ = np.polyfit(np.arange(len(pole), dtype=float), pole, 1)
            pole_height   = abs(float(pole[-1]) - float(pole[0]))
            confidence    = _clamp(1.0 - rest_range / (pole_range * 0.5))
            if pole_slope > 0:
                return PatternResult(
                    name="強気フラッグ",
                    direction="up",
                    forecast_prices=_make_target_forecast(last_close, last_close + pole_height, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )
            else:
                return PatternResult(
                    name="弱気フラッグ",
                    direction="down",
                    forecast_prices=_make_target_forecast(last_close, last_close - pole_height, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 11. カップアンドハンドル: U字カップ + 右端の小幅ハンドル下落（60本以上）
    if len(close) >= 60:
        n_c = len(close)
        cup_left  = close[:n_c // 3]
        cup_mid   = close[n_c // 3 : 2 * n_c // 3]
        cup_right = close[2 * n_c // 3:]
        left_mean  = float(cup_left.mean())
        mid_mean   = float(cup_mid.mean())
        right_mean = float(cup_right.mean())
        if left_mean > 0 and left_mean > mid_mean and right_mean > mid_mean:
            cup_depth = (left_mean - mid_mean) / left_mean
            if cup_depth >= 0.03:
                right_max    = float(cup_right.max())
                last_c       = float(cup_right[-1])
                handle_dip   = (right_max - last_c) / right_max if right_max > 0 else 1.0
                if handle_dip <= 0.05:
                    cup_depth_abs = left_mean - mid_mean
                    target        = right_max + cup_depth_abs
                    confidence    = _clamp(1.0 - handle_dip / 0.05)
                    return PatternResult(
                        name="カップアンドハンドル",
                        direction="up",
                        forecast_prices=_make_target_forecast(last_close, target, n_steps),
                        forecast_index=forecast_index,
                        confidence=confidence,
                    )

    # 12. ダブルトップ: 極大 2 つ以上、価格差 ≤ 2%、間に極小が存在
    if len(maxima) >= 2:
        p1, p2 = maxima[-2], maxima[-1]
        v1, v2 = close[p1], close[p2]
        peak_diff = abs(v1 - v2) / max(v1, v2)
        if peak_diff <= 0.02:
            between_minima = minima[(minima > p1) & (minima < p2)]
            if len(between_minima) > 0:
                neckline   = float(close[between_minima].min())
                peak_avg   = (v1 + v2) / 2.0
                target     = neckline - (peak_avg - neckline)
                confidence = _clamp(1.0 - peak_diff / 0.02)
                return PatternResult(
                    name="ダブルトップ",
                    direction="down",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    # 13. ダブルボトム: 極小 2 つ以上、価格差 ≤ 2%、間に極大が存在
    if len(minima) >= 2:
        t1, t2 = minima[-2], minima[-1]
        v1, v2 = close[t1], close[t2]
        trough_diff = abs(v1 - v2) / max(v1, v2)
        if trough_diff <= 0.02:
            between_maxima = maxima[(maxima > t1) & (maxima < t2)]
            if len(between_maxima) > 0:
                neckline   = float(close[between_maxima].max())
                trough_avg = (v1 + v2) / 2.0
                target     = neckline + (neckline - trough_avg)
                confidence = _clamp(1.0 - trough_diff / 0.02)
                return PatternResult(
                    name="ダブルボトム",
                    direction="up",
                    forecast_prices=_make_target_forecast(last_close, target, n_steps),
                    forecast_index=forecast_index,
                    confidence=confidence,
                )

    return None


def draw_prediction(ax, result: PatternResult, n_existing: int) -> None:
    """予測ラインとパターン名（信頼度付き）を axes に描画する。

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
    label = f"{result.name} ({result.confidence:.0%})"
    ax.text(
        0.98, 0.97,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#e67e22",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
