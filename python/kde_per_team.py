import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.stats import gaussian_kde

k = 0
grid = 4
shot_num = 13

# Parameters
HOUSE_CENTER = (0, 38.405)
HOUSE_RADIUS = 1.829
BOUNDS_X = (-2.5, 2.5)
BOUNDS_Y = (HOUSE_CENTER[1] - 2.5, HOUSE_CENTER[1] + 2.5)
RESOLUTION = 300j  # grid resolution


def draw_house(ax):
    ring_radii = [1.829, 1.219, 0.610]
    ring_colors = ["#D8ECF3", "#A8D1E7", "#6AB3DB"]

    for radius, color in zip(ring_radii, ring_colors):
        ring = plt.Circle(
            HOUSE_CENTER, radius, color=color, ec="black", lw=0.5, zorder=0
        )
        ax.add_patch(ring)

    button = plt.Circle(HOUSE_CENTER, 0.15, color="red", ec="black", lw=0.5, zorder=1)
    ax.add_patch(button)

    ax.set_aspect("equal")
    ax.set_xlim(BOUNDS_X)
    ax.set_ylim(BOUNDS_Y)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)


def plot_team_kde_heatmap(cluster_folder, output_base_path):
    team_coords = {0: ([], []), 1: ([], [])}
    files = glob.glob(os.path.join(cluster_folder, "*.csv"))

    for csv_path in files:
        df = pd.read_csv(csv_path)
        for team in range(2):
            for i in range(8):
                x = df.iloc[0, team * 16 + i * 2]
                y = df.iloc[0, team * 16 + i * 2 + 1]
                if pd.notna(x) and pd.notna(y):
                    team_coords[team][0].append(x)
                    team_coords[team][1].append(y)

    for team in [0, 1]:
        x, y = team_coords[team]
        if len(x) < 5:
            print(f"Not enough stones for Team {team} in {cluster_folder}")
            continue

        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        xi, yi = np.mgrid[
            BOUNDS_X[0] : BOUNDS_X[1] : RESOLUTION,
            BOUNDS_Y[0] : BOUNDS_Y[1] : RESOLUTION,
        ]
        coords = np.vstack([xi.ravel(), yi.ravel()])
        zi = np.reshape(kernel(coords), xi.shape)

        fig, ax = plt.subplots(figsize=(8, 8))
        draw_house(ax)

        cmap = "Reds" if team == 0 else "Blues"
        label = f"Team {team} Stone Density"

        heatmap = ax.imshow(
            zi.T,
            extent=[*BOUNDS_X, *BOUNDS_Y],
            origin="lower",
            cmap=cmap,
            alpha=0.6,
            zorder=0.5,
            interpolation="bilinear",
        )
        plt.colorbar(heatmap, ax=ax, label=label)
        ax.set_title(label)
        ax.text(
            0.97,
            0.03,
            f"Total Stones: {len(x)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2"),
        )

        save_path = f"{output_base_path}_Team{team}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400)
        plt.close()
        print(f"Saved Team {team} heatmap to: {save_path}")


plot_team_kde_heatmap(
    cluster_folder=f"../build/Release/hierarchical_clustering/Stone_Coordinates_{grid}_{grid}/shot{shot_num}/Cluster{k}",
    output_base_path=f"../images/StonePlots/kde_cluster{k}_shot{shot_num}",
)
