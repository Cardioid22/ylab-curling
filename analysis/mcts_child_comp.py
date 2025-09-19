import pandas as pd
import os
import re
from glob import glob
import math
import sys

# --- Parameters ---
if len(sys.argv) < 2:
    print("Usage: python a.py <grid_size>")
    sys.exit(1)

grid_size = int(sys.argv[1])
cluster_num = int(math.log2(grid_size * grid_size))
is_first = True  # 偶数(先手)ならTrue, 奇数(後手)ならFalse

# --- Paths ---
base_dir = f"../remote_log/Grid_{grid_size}x{grid_size}/"
comparison_dir = os.path.join(base_dir, "MCTS_Output_BestShotComparison_")
cluster_dir = os.path.join(base_dir, f"MCTS_Output_ClusteringId_{cluster_num}_Clusters")

# --- Collect comparison files ---
comparison_files = sorted(
    glob(os.path.join(comparison_dir, "best_shot_comparison_*.csv"))
)

odd_comparison_files = [
    f
    for f in comparison_files
    if re.search(r"best_shot_comparison_(\d+)\.csv", os.path.basename(f))
    and int(re.search(r"best_shot_comparison_(\d+)\.csv", os.path.basename(f)).group(1))
    % 2
    == 1
]
even_comparison_files = [
    f
    for f in comparison_files
    if re.search(r"best_shot_comparison_(\d+)\.csv", os.path.basename(f))
    and int(re.search(r"best_shot_comparison_(\d+)\.csv", os.path.basename(f)).group(1))
    % 2
    == 0
]

selected_comparison_files = even_comparison_files if is_first else odd_comparison_files

# --- Collect cluster files ---
cluster_files = sorted(glob(os.path.join(cluster_dir, "cluster_ids_*.csv")))

odd_cluster_files = [
    f
    for f in cluster_files
    if re.search(r"cluster_ids_(\d+)\.csv", os.path.basename(f))
    and int(re.search(r"cluster_ids_(\d+)\.csv", os.path.basename(f)).group(1)) % 2 == 1
]
even_cluster_files = [
    f
    for f in cluster_files
    if re.search(r"cluster_ids_(\d+)\.csv", os.path.basename(f))
    and int(re.search(r"cluster_ids_(\d+)\.csv", os.path.basename(f)).group(1)) % 2 == 0
]

selected_cluster_files = even_cluster_files if is_first else odd_cluster_files

# --- Load All Cluster Files ---
cluster_map = {}  # state_id -> cluster_id
for cluster_file in selected_cluster_files:
    try:
        cluster_df = pd.read_csv(cluster_file)
        for _, row in cluster_df.iterrows():
            cluster_map[int(row["StateId"])] = int(row["ClusterId"])
    except Exception as e:
        print(f"Failed to load cluster file {cluster_file}: {e}")
print(f"Loaded {len(selected_cluster_files)} cluster files")

# --- Analyze Each Comparison File ---
results = []

for file in selected_comparison_files:
    try:
        df = pd.read_csv(file)
        mcts_row = df[df["Type"] == "MCTS"]
        grid_row = df[df["Type"] == "AllGrid"]

        if mcts_row.empty or grid_row.empty:
            print(f"Missing row in {file}")
            continue

        mcts_id = int(mcts_row["StateID"].values[0])
        grid_id = int(grid_row["StateID"].values[0])

        mcts_cluster = cluster_map.get(mcts_id, None)
        grid_cluster = cluster_map.get(grid_id, None)

        same_cluster = (mcts_cluster == grid_cluster) and mcts_cluster is not None

        results.append(
            {
                "File": os.path.basename(file),
                "MCTS_State": mcts_id,
                "AllGrid_State": grid_id,
                "MCTS_Cluster": mcts_cluster,
                "AllGrid_Cluster": grid_cluster,
                "SameCluster": same_cluster,
            }
        )
    except Exception as e:
        print(f"Failed to process {file}: {e}")

print(f"Processed {len(selected_comparison_files)} comparison files")

# --- Save Summary ---
output_file = (
    os.path.join(comparison_dir, "best_shot_cluster_similarity_first.csv")
    if is_first
    else os.path.join(comparison_dir, "best_shot_cluster_similarity_last.csv")
)
summary_df = pd.DataFrame(results)
summary_df.to_csv(output_file, index=False)

print(f"\n✅ Summary saved to: {output_file}")
# print(summary_df.head())
