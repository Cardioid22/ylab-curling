import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

grid = 8
input_folder = (
    f"build/Release/hierarchical_clustering/cluster_distribution_{grid}_{grid}/"
)
output_folder = f"hierarchical_clustering/output_distribution_{grid}_{grid}"
os.makedirs(output_folder, exist_ok=True)

cmap = plt.get_cmap("tab10")


def load_cluster_grid(filename):
    df = pd.read_csv(os.path.join(input_folder, filename), header=None)
    return df.to_numpy(dtype=int)


def save_cluster_distribution_plot(filename):
    grid_data = load_cluster_grid(filename)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid_data, cmap=cmap, interpolation="none")

    for x in range(grid_data.shape[1] + 1):
        ax.axvline(x - 0.5, color="black", linewidth=0.8)
    for y in range(grid_data.shape[0] + 1):
        ax.axhline(y - 0.5, color="black", linewidth=0.8)

    for i in range(grid_data.shape[0]):
        for j in range(grid_data.shape[1]):
            ax.text(
                j,
                i,
                grid_data[i, j],
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    ax.set_title(filename)
    ax.set_xticks([])
    ax.set_yticks([])

    # plt.colorbar(im, ax=ax, shrink=0.8, label="Cluster ID")
    plt.tight_layout()

    output_path = os.path.join(output_folder, filename.replace(".csv", ".png"))
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def save_combined_image(
    filenames, combined_filename="combined_cluster_distributions.png"
):
    n = len(filenames)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, filename in enumerate(filenames):
        grid_data = load_cluster_grid(filename)
        ax = axes[idx]
        im = ax.imshow(grid_data, cmap=cmap, interpolation="none")

        for x in range(grid_data.shape[1] + 1):
            ax.axvline(x - 0.5, color="black", linewidth=0.8)
        for y in range(grid_data.shape[0] + 1):
            ax.axhline(y - 0.5, color="black", linewidth=0.8)

        for i in range(grid_data.shape[0]):
            for j in range(grid_data.shape[1]):
                ax.text(
                    j,
                    i,
                    grid_data[i, j],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )

        ax.set_title(filename)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n, len(axes)):
        axes[i].axis("off")

    # fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Cluster ID")
    plt.tight_layout()

    combined_path = os.path.join(output_folder, combined_filename)
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"Combined image saved to: {combined_path}")


# Process all CSV files
files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])
for f in files:
    save_cluster_distribution_plot(f)

# Save combined image
save_combined_image(files)
