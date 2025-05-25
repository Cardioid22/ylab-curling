import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.stats import gaussian_kde

k = 3
grid = 4
shot_num = 13

# Parameters
HOUSE_CENTER = (0, 38.405)
HOUSE_RADIUS = 1.829
BOUNDS_X = (-2.5, 2.5)
BOUNDS_Y = (HOUSE_CENTER[1] - 2.5, HOUSE_CENTER[1] + 2.5)
RESOLUTION = 300j  # grid resolution


def draw_house(ax):
    # Draw curling rings
    ring_radii = [1.829, 1.219, 0.610]  # 6ft, 4ft, 2ft
    ring_colors = ["#D8ECF3", "#A8D1E7", "#6AB3DB"]  # lighter blue shades

    for radius, color in zip(ring_radii, ring_colors):
        ring = plt.Circle(
            HOUSE_CENTER, radius, color=color, ec="black", lw=0.5, zorder=0
        )
        ax.add_patch(ring)

    # Red center button
    button = plt.Circle(HOUSE_CENTER, 0.15, color="red", ec="black", lw=0.5, zorder=1)
    ax.add_patch(button)

    ax.set_aspect("equal")
    ax.set_xlim(BOUNDS_X)
    ax.set_ylim(BOUNDS_Y)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Stone Density Heatmap, end: {shot_num}, cluster:{k}")
    ax.grid(True)


def plot_cluster_kde_heatmap(cluster_folder, output_path):
    all_x, all_y = [], []
    files = glob.glob(os.path.join(cluster_folder, "*.csv"))

    for csv_path in files:
        df = pd.read_csv(csv_path)
        for team in range(2):
            for i in range(8):
                x = df.iloc[0, team * 16 + i * 2]
                y = df.iloc[0, team * 16 + i * 2 + 1]
                if pd.notna(x) and pd.notna(y):
                    all_x.append(x)
                    all_y.append(y)

    if len(all_x) < 5:
        print(
            f"Not enough data points to create a meaningful KDE heatmap in {cluster_folder}"
        )
        return

    # KDE estimate
    values = np.vstack([all_x, all_y])
    kernel = gaussian_kde(values)
    xi, yi = np.mgrid[
        BOUNDS_X[0] : BOUNDS_X[1] : RESOLUTION, BOUNDS_Y[0] : BOUNDS_Y[1] : RESOLUTION
    ]
    coords = np.vstack([xi.ravel(), yi.ravel()])
    zi = np.reshape(kernel(coords), xi.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_house(ax)

    heatmap = ax.imshow(
        zi.T,
        extent=[*BOUNDS_X, *BOUNDS_Y],
        origin="lower",
        cmap="coolwarm",  # Better contrast
        alpha=0.6,
        zorder=0.5,
        interpolation="bilinear",
    )
    plt.colorbar(heatmap, ax=ax, label="Estimated Stone Density")
    ax.text(
        0.97,
        0.03,
        f"Total Stones: {len(all_x)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2"),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=400)
    plt.close()
    print(f"Saved KDE heatmap to: {output_path}")


plot_cluster_kde_heatmap(
    cluster_folder=f"../build/Release/hierarchical_clustering/Stone_Coordinates_{grid}_{grid}/shot{shot_num}/Cluster{k}",
    output_path=f"../images/StonePlots/heatmap_shot{shot_num}_Cluster{k}.png",
)
