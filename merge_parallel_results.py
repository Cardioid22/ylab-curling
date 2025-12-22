#!/usr/bin/env python3
"""
Merge Parallel Agreement Experiment Results

This script merges multiple CSV files from parallel agreement experiment
into a single combined CSV file for easier analysis.

Usage:
    python merge_parallel_results.py <result_directory>

Example:
    python merge_parallel_results.py experiments/parallel_agreement_results/Grid16_Depth1_Clusters4_Tests100_20251222143000
"""

import pandas as pd
import sys
from pathlib import Path
import glob


def merge_results(result_dir: Path):
    """Merge all test result CSVs into a single file"""

    # Find all test result CSV files
    csv_pattern = str(result_dir / "test_*_result.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"Error: No test result CSV files found in {result_dir}")
        print(f"Pattern: {csv_pattern}")
        return

    print(f"Found {len(csv_files)} test result files")
    print(f"Merging into single CSV...")

    # Read and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")

    if not dfs:
        print("Error: No valid CSV files could be read")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by TestID and Method
    combined_df = combined_df.sort_values(['TestID', 'Method'])

    # Save combined results
    output_file = result_dir / "combined_results.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"\nMerged {len(dfs)} files successfully")
    print(f"Total rows: {len(combined_df)}")
    print(f"Output: {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    # Count unique test IDs
    unique_tests = combined_df['TestID'].nunique()
    print(f"Unique test cases: {unique_tests}")

    # Separate AllGrid and Clustered results
    allgrid_df = combined_df[combined_df['Method'] == 'AllGrid']
    clustered_df = combined_df[combined_df['Method'] == 'Clustered']

    print(f"AllGrid results: {len(allgrid_df)}")
    print(f"Clustered results: {len(clustered_df)}")

    # Calculate agreement rates by iteration count
    if 'ClusterAgreement' in clustered_df.columns:
        print("\nAgreement Rates by Iteration Count:")
        print("-" * 60)

        iteration_groups = clustered_df.groupby('Iterations')

        for iterations, group in iteration_groups:
            total = len(group)
            exact_agree = (group['ExactAgreement'] == 'YES').sum()
            cluster_agree = (group['ClusterAgreement'] == 'YES').sum()

            exact_rate = (exact_agree / total * 100) if total > 0 else 0
            cluster_rate = (cluster_agree / total * 100) if total > 0 else 0

            print(f"  Iterations {iterations:6d}: "
                  f"Exact={exact_rate:5.1f}% ({exact_agree:3d}/{total:3d}), "
                  f"Cluster={cluster_rate:5.1f}% ({cluster_agree:3d}/{total:3d})")

    # Average silhouette score
    if 'SilhouetteScore' in clustered_df.columns:
        avg_silhouette = clustered_df['SilhouetteScore'].mean()
        print(f"\nAverage Silhouette Score: {avg_silhouette:.4f}")

    # Average timing
    if 'ElapsedTime' in combined_df.columns:
        avg_allgrid_time = allgrid_df['ElapsedTime'].mean()
        avg_clustered_time = clustered_df['ElapsedTime'].mean()

        print(f"\nAverage Execution Time:")
        print(f"  AllGrid: {avg_allgrid_time:.2f}s")
        print(f"  Clustered (avg across all iterations): {avg_clustered_time:.2f}s")

        if avg_allgrid_time > 0:
            speedup = avg_allgrid_time / avg_clustered_time
            print(f"  Average speedup: {speedup:.2f}x")

    print("="*60)

    # Export cluster details if available
    cluster_detail_files = sorted(glob.glob(str(result_dir / "test_*_all_clusters_details.csv")))
    if cluster_detail_files:
        print(f"\nMerging {len(cluster_detail_files)} cluster detail files...")

        detail_dfs = []
        for csv_file in cluster_detail_files:
            try:
                df = pd.read_csv(csv_file)
                detail_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to read {csv_file}: {e}")

        if detail_dfs:
            combined_details = pd.concat(detail_dfs, ignore_index=True)
            output_detail_file = result_dir / "combined_cluster_details.csv"
            combined_details.to_csv(output_detail_file, index=False)
            print(f"Cluster details saved to: {output_detail_file}")

    # Export best shot comparisons if available
    best_shot_files = sorted(glob.glob(str(result_dir / "test_*_best_shot_comparison.csv")))
    if best_shot_files:
        print(f"\nMerging {len(best_shot_files)} best shot comparison files...")

        best_shot_dfs = []
        for csv_file in best_shot_files:
            try:
                df = pd.read_csv(csv_file)
                best_shot_dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to read {csv_file}: {e}")

        if best_shot_dfs:
            combined_best_shots = pd.concat(best_shot_dfs, ignore_index=True)
            output_best_shot_file = result_dir / "combined_best_shot_comparison.csv"
            combined_best_shots.to_csv(output_best_shot_file, index=False)
            print(f"Best shot comparisons saved to: {output_best_shot_file}")

    print("\nMerge complete!")


def main():
    if len(sys.argv) != 2:
        print("Usage: python merge_parallel_results.py <result_directory>")
        print("\nExample:")
        print("  python merge_parallel_results.py experiments/parallel_agreement_results/Grid16_Depth1_Clusters4_Tests100_20251222143000")
        sys.exit(1)

    result_dir = Path(sys.argv[1])

    if not result_dir.exists():
        print(f"Error: Directory does not exist: {result_dir}")
        sys.exit(1)

    if not result_dir.is_dir():
        print(f"Error: Not a directory: {result_dir}")
        sys.exit(1)

    print(f"Merging results from: {result_dir}")
    print("")

    merge_results(result_dir)


if __name__ == "__main__":
    main()
