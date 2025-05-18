import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

grid = 8
folder = f"build/Release/hierarchical_clustering/Intra_Cluster_Scores_{grid}_{grid}/"
pattern = os.path.join(folder, "intra_cluster_scores_shot_*.csv")


def extract_shot_number(filepath):
    match = re.search(r"shot_(\d+)", filepath)
    return int(match.group(1)) if match else -1


files = sorted(glob.glob(pattern), key=extract_shot_number)

# Plot each file
plt.figure(figsize=(10, 6))
for file_path in files:
    data = pd.read_csv(file_path)
    if "k" in data.columns and "intra_score" in data.columns:
        shot_num = os.path.splitext(os.path.basename(file_path))[0].split("_")[-1]
        plt.plot(data["k"], data["intra_score"], marker="o", label=f"Shot {shot_num}")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Intra-cluster Distance")
plt.title("Elbow Method Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
