import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 設定 (ご自身の環境に合わせて変更してください) ---

# グリッドサイズ (パスの構築にのみ使用)
if len(sys.argv) < 2:
    print("Usage: python a.py <grid_size>")
    sys.exit(1)

n = int(sys.argv[1])

# 解析対象のCSVファイル名
csv_filename = "best_shot_cluster_similarity_first.csv"

# ベースディレクトリと出力ディレクトリのパス
base_dir = Path(f"../remote_log/Grid_{n}x{n}/MCTS_Output_BestShotComparison_")
out_base = Path(f"../images/MCTS_Analysis/Grid_{n}_{n}")

# --- ここから下は自動処理 ---


def main():
    """
    CSVファイルを読み込み、'SameCluster' 列の TRUE/FALSE を集計して
    色付き棒グラフを生成・保存するメイン関数。
    """
    # 1. パスの構築
    input_csv_path = base_dir / csv_filename

    # 出力ディレクトリが存在しない場合は作成
    out_base.mkdir(parents=True, exist_ok=True)

    # 2. CSVファイルの読み込み
    try:
        df = pd.read_csv(input_csv_path)
        print(f"✅ ファイルを正常に読み込みました: {input_csv_path}")
    except FileNotFoundError:
        print(
            f"❌ エラー: ファイルが見つかりません。パスを確認してください: {input_csv_path}"
        )
        return
    except Exception as e:
        print(f"❌ ファイル読み込み中にエラーが発生しました: {e}")
        return

    # 3. 'SameCluster' 列の存在チェック
    if "SameCluster" not in df.columns:
        print(
            f"❌ エラー: CSVファイルに 'SameCluster' という名前の列が見つかりません。"
        )
        return

    # 4. TRUE と FALSE の個数を集計
    # value_counts() を使うと、各値の出現回数を簡単に集計できます。
    # 大文字小文字の違いに対応するため、文字列に変換して小文字にする
    s_col = df["SameCluster"].astype(str).str.lower()
    counts = s_col.value_counts()

    # 特定のカテゴリのみを扱う（'true'と'false'）
    filtered_counts = {}
    if "true" in counts:
        filtered_counts["TRUE"] = counts["true"]
    if "false" in counts:
        filtered_counts["FALSE"] = counts["false"]

    if not filtered_counts:
        print(
            "❌ 'SameCluster' 列に 'TRUE' または 'FALSE' のデータが見つかりませんでした。"
        )
        return

    # 描画のためにDataFrameに変換（順序を保証するため）
    plot_data = pd.Series(filtered_counts).sort_index(
        ascending=False
    )  # 'TRUE'を左、'FALSE'を右にするため

    print("\n--- 集計結果 ---")
    print(plot_data)
    print("----------------\n")

    # 5. グラフの描画
    plt.style.use("seaborn-v0_8-whitegrid")  # グラフのスタイルを設定
    fig, ax = plt.subplots(figsize=(8, 6))  # グラフのサイズを決定

    # 各カテゴリに対応する色を定義
    # より直感的な色に調整
    colors = {"TRUE": "#4CAF50", "FALSE": "#FF5722"}  # 緑系  # 赤系

    # plot_dataのインデックス（'TRUE', 'FALSE'）に対応する色を割り当てる
    bar_colors = [colors.get(label, "#cccccc") for label in plot_data.index]

    # 棒グラフを作成
    plot_data.plot(kind="bar", ax=ax, color=bar_colors, zorder=2)

    # グラフの各種設定
    ax.set_title(f'Count of TRUE/FALSE in "{csv_filename}"', fontsize=16)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Category", fontsize=12)
    ax.tick_params(axis="x", rotation=0)  # X軸のラベルを水平にする

    # 棒の上に数値を表示
    for i, count in enumerate(plot_data):
        ax.text(
            i,
            count + (plot_data.max() * 0.01),
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()  # レイアウトを自動調整

    # 6. グラフを画像ファイルとして保存
    output_path = (
        out_base / f"cluster_comparison_results_{n}_{n}.png"
    )  # ファイル名を変更
    plt.savefig(output_path)

    print(f"✅ 色付きグラフを保存しました: {output_path}")
    plt.close()  # メモリ解放のためプロットを閉じる


if __name__ == "__main__":
    main()
