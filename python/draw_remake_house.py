import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# Constants
GRID = 4
HOUSE_CENTER = (0, 38.405)
HOUSE_RADII = [1.829, 1.219, 0.610]  # 6ft, 4ft, 2ft rings
INPUT_ROOT = f"../build/Release/hierarchical_clustering/Stone_Coordinates_{GRID}_{GRID}"
OUTPUT_ROOT = f"../images/hierarchical_clustering/output_coordinate_{GRID}_{GRID}"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Constants
HOUSE_CENTER = (0, 38.405)
HOUSE_RADII = [1.829, 1.219, 0.610]  # 6ft, 4ft, 2ft in meters
COLORS = ["blue", "white", "red"]  # 12ft, 8ft, 4ft rings
BUTTON_RADIUS = 0.15  # Small red button in the center


def draw_house(ax):
    # Draw house rings from largest to smallest
    for r, color in zip(HOUSE_RADII, COLORS):
        ring = plt.Circle(
            HOUSE_CENTER, r, facecolor=color, edgecolor="black", linewidth=1.2, zorder=0
        )
        ax.add_patch(ring)

    # Draw button (center)
    button = plt.Circle(
        HOUSE_CENTER,
        BUTTON_RADIUS,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(button)

    # Axes settings
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(HOUSE_CENTER[1] - 2.5, HOUSE_CENTER[1] + 2.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Stone Positions on Curling Sheet")
    ax.grid(True)


def plot_stone_state(csv_path, output_path):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_house(ax)

    # Plot stones with visible edges
    colors = ["red", "blue"]
    for team in range(2):
        for i in range(8):
            x = df.iloc[0, team * 16 + i * 2]
            y = df.iloc[0, team * 16 + i * 2 + 1]
            if pd.notna(x) and pd.notna(y):
                ax.plot(
                    x,
                    y,
                    "o",
                    color=colors[team],
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label=f"Team{team}" if i == 0 else "",
                )

    ax.legend()
    ax.set_title(os.path.basename(csv_path).replace(".csv", ""))
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1


def process_all_states():
    for shot_dir in sorted(os.listdir(INPUT_ROOT), key=extract_number):
        shot_path = os.path.join(INPUT_ROOT, shot_dir)
        if not os.path.isdir(shot_path):
            continue
        for cluster_dir in sorted(os.listdir(shot_path), key=extract_number):
            cluster_path = os.path.join(shot_path, cluster_dir)
            if not os.path.isdir(cluster_path):
                continue
            for csv_file in sorted(os.listdir(cluster_path), key=extract_number):
                if not csv_file.endswith(".csv"):
                    continue
                input_csv = os.path.join(cluster_path, csv_file)
                relative_path = os.path.join(
                    shot_dir, cluster_dir, csv_file.replace(".csv", ".png")
                )
                output_path = os.path.join(OUTPUT_ROOT, relative_path)
                plot_stone_state(input_csv, output_path)


# Run
process_all_states()
