import pandas as pd
import matplotlib.pyplot as plt
import os

grid = 8
cluster = 8
folder = f"../build/Release/hierarchical_clustering/SilhouetteScores_{grid}_{grid}/"
file_path = os.path.join(folder, f"silhouette_scores_cluster_{cluster}.csv")

# Load the single CSV file
data = pd.read_csv(file_path)

# Sort by shot number (just in case)
data = data.sort_values(by="shot")

# Plot the silhouette score over shots
plt.figure(figsize=(10, 6))
plt.plot(data["shot"], data["silhouette_score"], marker="o", linestyle="-")
plt.xlabel("Shot Number")
plt.ylabel("Silhouette Score")
plt.title(f"Silhouette Score Across Shots (Cluster: {cluster})")
plt.grid(True)
plt.tight_layout()
plt.show()
