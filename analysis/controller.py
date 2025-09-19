import subprocess

# --- 共通の grid_size をここで指定 ---
grid_size = 70  # 例: 4x4

# --- 実行するプログラム ---
scripts = ["mcts_rollout_report.py", "mcts_child_comp.py", "mcts_plot_cluster_diff.py"]

# --- 各プログラムを順番に実行 ---
for script in scripts:
    print(f"Running {script} with grid_size={grid_size}...")
    subprocess.run(["python", script, str(grid_size)], check=True)

print("\n✅ All scripts finished successfully.")
