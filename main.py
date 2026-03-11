#!/usr/bin/env python3
"""Yahoo!ファイナンス 銘柄情報取得 CLI

Usage:
    python main.py <銘柄コード> [--chart] [--interval SEC] [--predict]

Arguments:
    code            東証銘柄コード (例: 7203)

Options:
    --chart         当日1分足のローソク足チャートをGUIウィンドウで表示する
    --interval SEC  チャートを N 秒ごとに自動更新する (--chart と併用)
    --predict       チャートパターンを検出して将来予測ラインをオーバーレイ表示する

Examples:
    # 株価情報をCUIで表示
    python main.py 7203

    # ローソク足チャートを表示
    python main.py 7203 --chart

    # 60秒ごとに自動更新しながらチャートを表示
    python main.py 7203 --chart --interval 60

    # チャートにパターン予測オーバーレイを表示
    python main.py 7203 --chart --predict

    # 自動更新 + 予測（毎フレーム再検出）
    python main.py 7203 --chart --interval 60 --predict
"""
import argparse
import sys

from src.fetcher import fetch_stock_info, fetch_intraday_data, StockNotFoundError, FetchError
from src.display import print_stock_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="指定した銘柄コードの株価情報を Yahoo! Finance から取得して表示します。"
    )
    parser.add_argument("code", help="東証銘柄コード (例: 7203)")
    parser.add_argument(
        "--chart",
        action="store_true",
        help="当日1分足のローソク足チャートをGUIウィンドウで表示する",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        metavar="SEC",
        help="チャートを N 秒ごとに自動更新する（例: --chart --interval 60）",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="チャートパターンを検出して将来予測ラインをオーバーレイ表示する",
    )
    args = parser.parse_args()

    try:
        info = fetch_stock_info(args.code)
    except StockNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FetchError as e:
        print(str(e), file=sys.stderr)
        return 1

    print_stock_info(info)

    if args.chart:
        from src.chart import plot_intraday, plot_intraday_live

        if args.interval is not None:
            try:
                plot_intraday_live(args.code, info=info, interval_sec=args.interval, predict=args.predict)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 1
            except Exception as e:
                print(
                    f"グラフ表示に失敗しました。環境変数 MPLBACKEND=TkAgg を設定して再実行してください。\n({e})",
                    file=sys.stderr,
                )
                return 1
        else:
            try:
                df = fetch_intraday_data(args.code)
            except FetchError as e:
                print(str(e), file=sys.stderr)
                return 1

            if df.empty:
                print(f"当日の取引データがありません: {args.code}", file=sys.stderr)
                return 1

            try:
                plot_intraday(df, info=info, predict=args.predict)
            except Exception as e:
                print(
                    f"グラフ表示に失敗しました。環境変数 MPLBACKEND=TkAgg を設定して再実行してください。\n({e})",
                    file=sys.stderr,
                )
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
