import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
import re
import sys

# --- Parameters ---
if len(sys.argv) < 2:
    print("Usage: python a.py <grid_size>")
    sys.exit(1)

grid_size = int(sys.argv[1])
base_dir = f"../remote_log/Grid_{grid_size}x{grid_size}/"
output_dir = "../images/MCTS_Analysis/"
os.makedirs(output_dir, exist_ok=True)


# --- Detect latest Iter folder for each kind ---
def find_latest_iter(kind: str) -> int:
    pattern = re.compile(r"Iter_(\d+)")
    iter_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Iter_")
    ]

    iter_candidates = []
    for d in iter_dirs:
        iter_num = int(pattern.match(d).group(1))
        files = glob(os.path.join(base_dir, d, f"root_children_score_{kind}_*.csv"))
        if files:
            iter_candidates.append(iter_num)

    if not iter_candidates:
        raise FileNotFoundError(f"No Iter folders found for kind '{kind}'")

    return max(iter_candidates)


def find_all_shots(iter_dir: str, kind: str):
    # ファイル名の末尾番号部分からShot番号を抽出
    files = glob(os.path.join(iter_dir, f"root_children_score_{kind}_*.csv"))
    shot_nums = []
    for f in files:
        m = re.search(rf"root_children_score_{kind}_(\d+)\.csv$", os.path.basename(f))
        if m:
            shot_nums.append(int(m.group(1)))
    return sorted(set(shot_nums))


# detect iterations
clustered_iter = find_latest_iter("clustered")
allgrid_iter = find_latest_iter("allgrid")
print(f"Detected clustered_iter={clustered_iter}, allgrid_iter={allgrid_iter}")

# detect shot numbers from whichever kind has them
clustered_shots = find_all_shots(
    os.path.join(base_dir, f"Iter_{clustered_iter}"), "clustered"
)
allgrid_shots = find_all_shots(
    os.path.join(base_dir, f"Iter_{allgrid_iter}"), "allgrid"
)
shot_nums = sorted(set(clustered_shots + allgrid_shots))
print(f"Detected shot numbers: {shot_nums}")

# --- Load all relevant files ---
all_data = []
iter_map = {"clustered": clustered_iter, "allgrid": allgrid_iter}

for kind, iter_num in iter_map.items():
    iter_dir = os.path.join(base_dir, f"Iter_{iter_num}")
    for shot_num in shot_nums:
        file_path = os.path.join(iter_dir, f"root_children_score_{kind}_{shot_num}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Iteration"] = iter_num
                df["Shot"] = shot_num
                df["Type"] = kind.capitalize()
                all_data.append(df)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

# --- Combine all data ---
if not all_data:
    print("No data loaded.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# --- Summary & Plot ---
summary_stats = (
    combined_df.groupby(["Type", "Iteration"])["Score"].describe().reset_index()
)
print(summary_stats)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x="Type", y="Score", data=combined_df, palette="Set2")

output_path = os.path.join(
    output_dir,
    f"mcts_rollout_report_{clustered_iter}_{allgrid_iter}_{grid_size}x{grid_size}.png",
)
plt.title("MCTS Score Distribution Across All Shots and Iterations")
plt.xlabel("Node Type")
plt.ylabel("Score")

# Add stats text
stats_text = ""
for _, row in summary_stats.iterrows():
    type_ = row["Type"]
    iter_ = int(row["Iteration"])
    mean = row["mean"]
    std = row["std"]
    stats_text += f"Type: {type_}, Iter: {iter_}, Mean: {mean:.2f}, Std: {std:.2f}\n"

plt.gcf().text(0.95, 0.01, stats_text, fontsize=8, va="bottom", ha="right")
plt.tight_layout()
plt.savefig(output_path, dpi=400)
plt.show()
