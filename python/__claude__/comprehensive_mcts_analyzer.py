#!/usr/bin/env python3
"""
Comprehensive MCTS Performance Analyzer

This module provides detailed comparative analysis between Clustered MCTS and AllGrid MCTS algorithms,
analyzing performance metrics, efficiency, and strategic effectiveness across different game configurations.

Features:
- Performance comparison with statistical significance testing
- Computational efficiency vs accuracy tradeoff analysis
- Multi-grid scaling performance analysis
- Detailed visualization and reporting

Author: Auto-generated analysis tool for ylab-curling project
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMCTSAnalyzer:
    """Comprehensive analyzer for MCTS algorithm performance comparison."""

    def __init__(self, base_path: str = "../remote_log"):
        """
        Initialize the analyzer.

        Args:
            base_path: Path to the remote_log directory containing experiment results
        """
        self.base_path = base_path
        self.results = {}
        self.output_dir = "mcts_analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (12, 8),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def discover_experiments(self) -> Dict[str, Dict]:
        """
        Automatically discover and catalog all experiment directories.

        Returns:
            Dictionary containing experiment metadata organized by grid size and iteration count
        """
        experiments = {}

        if not os.path.exists(self.base_path):
            print(f"❌ Base path {self.base_path} does not exist!")
            return experiments

        # Pattern for different directory structures
        patterns = [
            r"Grid_(\d+)x(\d+)",
            r"Iter_(\d+)",
            r".*_(\d+)x(\d+).*",
            r".*Iterations.*"
        ]

        for root, dirs, files in os.walk(self.base_path):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)

                # Extract grid size and iteration info
                grid_match = re.search(r"(\d+)x(\d+)", dir_name)
                iter_match = re.search(r"Iter_(\d+)", dir_name) or re.search(r"(\d+)_Iterations", dir_name)

                if grid_match:
                    grid_size = f"{grid_match.group(1)}x{grid_match.group(2)}"
                    iterations = iter_match.group(1) if iter_match else "unknown"

                    exp_key = f"{grid_size}_{iterations}"
                    experiments[exp_key] = {
                        'path': full_path,
                        'grid_size': grid_size,
                        'iterations': iterations,
                        'clustered_files': [],
                        'allgrid_files': [],
                        'comparison_files': [],
                        'cluster_mapping_files': []
                    }

        # Find specific files in each experiment
        for exp_key, exp_data in experiments.items():
            exp_path = exp_data['path']

            # Find all CSV files recursively
            for csv_file in glob.glob(os.path.join(exp_path, "**", "*.csv"), recursive=True):
                filename = os.path.basename(csv_file)

                if "clustered" in filename:
                    exp_data['clustered_files'].append(csv_file)
                elif "allgrid" in filename:
                    exp_data['allgrid_files'].append(csv_file)
                elif "comparison" in filename:
                    exp_data['comparison_files'].append(csv_file)
                elif "cluster_ids" in filename:
                    exp_data['cluster_mapping_files'].append(csv_file)

        print(f"🔍 Discovered {len(experiments)} experiments")
        for exp_key, exp_data in experiments.items():
            print(f"  📁 {exp_key}: {len(exp_data['clustered_files'])} clustered, "
                  f"{len(exp_data['allgrid_files'])} allgrid files")

        return experiments

    def load_experiment_data(self, experiment: Dict) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a single experiment.

        Args:
            experiment: Experiment metadata dictionary

        Returns:
            Dictionary containing loaded DataFrames
        """
        data = {
            'clustered': [],
            'allgrid': [],
            'comparisons': [],
            'cluster_mappings': []
        }

        # Load clustered MCTS data
        for file_path in experiment['clustered_files']:
            try:
                df = pd.read_csv(file_path)
                shot_num = self._extract_shot_number(file_path)
                df['shot_number'] = shot_num
                df['algorithm'] = 'Clustered'
                data['clustered'].append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

        # Load AllGrid MCTS data
        for file_path in experiment['allgrid_files']:
            try:
                df = pd.read_csv(file_path)
                shot_num = self._extract_shot_number(file_path)
                df['shot_number'] = shot_num
                df['algorithm'] = 'AllGrid'
                data['allgrid'].append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

        # Load comparison data
        for file_path in experiment['comparison_files']:
            try:
                df = pd.read_csv(file_path)
                shot_num = self._extract_shot_number(file_path)
                df['shot_number'] = shot_num
                data['comparisons'].append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

        # Load cluster mapping data
        for file_path in experiment['cluster_mapping_files']:
            try:
                df = pd.read_csv(file_path)
                shot_num = self._extract_shot_number(file_path)
                df['shot_number'] = shot_num
                data['cluster_mappings'].append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")

        # Combine DataFrames
        combined_data = {}
        for key, df_list in data.items():
            if df_list:
                combined_data[key] = pd.concat(df_list, ignore_index=True)
            else:
                combined_data[key] = pd.DataFrame()

        return combined_data

    def _extract_shot_number(self, file_path: str) -> int:
        """Extract shot number from filename."""
        match = re.search(r"_(\d+)\.csv$", file_path)
        return int(match.group(1)) if match else 0

    def analyze_performance_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze core performance metrics for both algorithms.

        Args:
            data: Combined data from load_experiment_data

        Returns:
            Dictionary containing performance analysis results
        """
        results = {}

        # Combine clustered and allgrid data for comparison
        if not data['clustered'].empty and not data['allgrid'].empty:
            combined_mcts = pd.concat([data['clustered'], data['allgrid']], ignore_index=True)

            # Performance metrics by algorithm
            perf_by_algo = combined_mcts.groupby('algorithm').agg({
                'Score': ['mean', 'std', 'median', 'min', 'max'],
                'Visits': ['mean', 'std', 'median', 'sum'],
                'Wins': ['mean', 'std', 'sum'],
                'Vx': ['mean', 'std'],
                'Vy': ['mean', 'std']
            }).round(4)

            results['performance_by_algorithm'] = perf_by_algo

            # Performance metrics by shot number
            perf_by_shot = combined_mcts.groupby(['shot_number', 'algorithm']).agg({
                'Score': 'mean',
                'Visits': 'mean',
                'Wins': 'mean'
            }).reset_index()

            results['performance_by_shot'] = perf_by_shot

            # Win rate analysis
            win_rates = combined_mcts.groupby('algorithm').apply(
                lambda x: x['Wins'].sum() / x['Visits'].sum() if x['Visits'].sum() > 0 else 0
            )
            results['win_rates'] = win_rates

            # Statistical significance test
            clustered_scores = data['clustered']['Score'] if 'Score' in data['clustered'].columns else []
            allgrid_scores = data['allgrid']['Score'] if 'Score' in data['allgrid'].columns else []

            if len(clustered_scores) > 0 and len(allgrid_scores) > 0:
                stat, p_value = stats.mannwhitneyu(clustered_scores, allgrid_scores, alternative='two-sided')
                results['significance_test'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        return results

    def analyze_efficiency_tradeoffs(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze computational efficiency vs accuracy tradeoffs.

        Args:
            data: Combined data from load_experiment_data

        Returns:
            Dictionary containing efficiency analysis results
        """
        results = {}

        if not data['clustered'].empty and not data['allgrid'].empty:
            # Calculate efficiency metrics
            clustered_stats = data['clustered'].groupby('shot_number').agg({
                'Score': 'mean',
                'Visits': 'sum',
                'Wins': 'sum'
            })

            allgrid_stats = data['allgrid'].groupby('shot_number').agg({
                'Score': 'mean',
                'Visits': 'sum',
                'Wins': 'sum'
            })

            # Efficiency = Performance / Computational Cost
            clustered_efficiency = clustered_stats['Score'] / (clustered_stats['Visits'] + 1e-6)
            allgrid_efficiency = allgrid_stats['Score'] / (allgrid_stats['Visits'] + 1e-6)

            results['efficiency_by_shot'] = pd.DataFrame({
                'shot_number': clustered_efficiency.index,
                'clustered_efficiency': clustered_efficiency.values,
                'allgrid_efficiency': allgrid_efficiency.values
            })

            # Overall efficiency comparison
            results['overall_efficiency'] = {
                'clustered': clustered_efficiency.mean(),
                'allgrid': allgrid_efficiency.mean(),
                'efficiency_ratio': clustered_efficiency.mean() / (allgrid_efficiency.mean() + 1e-6)
            }

        return results

    def analyze_strategy_agreement(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze how often both algorithms agree on strategy.

        Args:
            data: Combined data from load_experiment_data

        Returns:
            Dictionary containing strategy agreement analysis
        """
        results = {}

        if not data['comparisons'].empty:
            comparisons = data['comparisons']

            # Calculate spatial distance between chosen shots
            mcts_shots = comparisons[comparisons['Type'] == 'MCTS']
            allgrid_shots = comparisons[comparisons['Type'] == 'AllGrid']

            if not mcts_shots.empty and not allgrid_shots.empty:
                merged = pd.merge(mcts_shots, allgrid_shots, on='shot_number', suffixes=('_mcts', '_allgrid'))

                # Euclidean distance in velocity space
                merged['velocity_distance'] = np.sqrt(
                    (merged['Vx_mcts'] - merged['Vx_allgrid'])**2 +
                    (merged['Vy_mcts'] - merged['Vy_allgrid'])**2
                )

                # Rotation agreement
                merged['rotation_agreement'] = merged['Rot_mcts'] == merged['Rot_allgrid']

                # State ID agreement (if available)
                if 'StateID_mcts' in merged.columns and 'StateID_allgrid' in merged.columns:
                    merged['state_agreement'] = merged['StateID_mcts'] == merged['StateID_allgrid']
                    results['state_agreement_rate'] = merged['state_agreement'].mean()

                results['velocity_distance_stats'] = {
                    'mean': merged['velocity_distance'].mean(),
                    'std': merged['velocity_distance'].std(),
                    'median': merged['velocity_distance'].median()
                }

                results['rotation_agreement_rate'] = merged['rotation_agreement'].mean()
                results['agreement_by_shot'] = merged.groupby('shot_number').agg({
                    'velocity_distance': 'mean',
                    'rotation_agreement': 'mean'
                })

        return results

    def create_performance_visualizations(self, experiment_results: Dict) -> None:
        """
        Create comprehensive performance visualization plots.

        Args:
            experiment_results: Results from analyze_experiment
        """
        # 1. Performance comparison by algorithm
        if 'performance_metrics' in experiment_results:
            metrics = experiment_results['performance_metrics']

            if 'performance_by_shot' in metrics:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                shot_data = metrics['performance_by_shot']

                # Score by shot number
                for algo in shot_data['algorithm'].unique():
                    algo_data = shot_data[shot_data['algorithm'] == algo]
                    axes[0,0].plot(algo_data['shot_number'], algo_data['Score'],
                                 marker='o', label=algo, linewidth=2)
                axes[0,0].set_title('Average Score by Shot Number')
                axes[0,0].set_xlabel('Shot Number')
                axes[0,0].set_ylabel('Score')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)

                # Visits by shot number
                for algo in shot_data['algorithm'].unique():
                    algo_data = shot_data[shot_data['algorithm'] == algo]
                    axes[0,1].plot(algo_data['shot_number'], algo_data['Visits'],
                                 marker='s', label=algo, linewidth=2)
                axes[0,1].set_title('Average Visits by Shot Number')
                axes[0,1].set_xlabel('Shot Number')
                axes[0,1].set_ylabel('Visits')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)

                # Win rate by shot number
                for algo in shot_data['algorithm'].unique():
                    algo_data = shot_data[shot_data['algorithm'] == algo]
                    axes[1,0].plot(algo_data['shot_number'], algo_data['Wins'],
                                 marker='^', label=algo, linewidth=2)
                axes[1,0].set_title('Average Wins by Shot Number')
                axes[1,0].set_xlabel('Shot Number')
                axes[1,0].set_ylabel('Wins')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)

                # Efficiency comparison
                if 'efficiency_analysis' in experiment_results:
                    eff_data = experiment_results['efficiency_analysis']['efficiency_by_shot']
                    axes[1,1].plot(eff_data['shot_number'], eff_data['clustered_efficiency'],
                                 marker='o', label='Clustered', linewidth=2)
                    axes[1,1].plot(eff_data['shot_number'], eff_data['allgrid_efficiency'],
                                 marker='s', label='AllGrid', linewidth=2)
                    axes[1,1].set_title('Efficiency by Shot Number')
                    axes[1,1].set_xlabel('Shot Number')
                    axes[1,1].set_ylabel('Score/Visits')
                    axes[1,1].legend()
                    axes[1,1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
                plt.close()

        # 2. Strategy agreement visualization
        if 'strategy_analysis' in experiment_results:
            strategy = experiment_results['strategy_analysis']

            if 'agreement_by_shot' in strategy:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                agreement_data = strategy['agreement_by_shot']

                # Velocity distance over shots
                axes[0].plot(agreement_data.index, agreement_data['velocity_distance'],
                           marker='o', linewidth=2, color='red')
                axes[0].set_title('Velocity Distance Between Algorithms')
                axes[0].set_xlabel('Shot Number')
                axes[0].set_ylabel('Euclidean Distance')
                axes[0].grid(True, alpha=0.3)

                # Rotation agreement over shots
                axes[1].plot(agreement_data.index, agreement_data['rotation_agreement'],
                           marker='s', linewidth=2, color='blue')
                axes[1].set_title('Rotation Agreement Between Algorithms')
                axes[1].set_xlabel('Shot Number')
                axes[1].set_ylabel('Agreement Rate')
                axes[1].set_ylim(0, 1)
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'strategy_agreement.png'))
                plt.close()

    def analyze_experiment(self, experiment: Dict) -> Dict:
        """
        Perform complete analysis on a single experiment.

        Args:
            experiment: Experiment metadata dictionary

        Returns:
            Dictionary containing all analysis results
        """
        print(f"🔬 Analyzing experiment: {experiment['grid_size']}_{experiment['iterations']}")

        # Load data
        data = self.load_experiment_data(experiment)

        if not any(not df.empty for df in data.values()):
            print("⚠️ No data found for this experiment")
            return {}

        results = {
            'experiment_info': experiment,
            'data_summary': {
                'clustered_samples': len(data['clustered']),
                'allgrid_samples': len(data['allgrid']),
                'comparison_samples': len(data['comparisons']),
                'cluster_mapping_samples': len(data['cluster_mappings'])
            }
        }

        # Perform analyses
        results['performance_metrics'] = self.analyze_performance_metrics(data)
        results['efficiency_analysis'] = self.analyze_efficiency_tradeoffs(data)
        results['strategy_analysis'] = self.analyze_strategy_agreement(data)

        return results

    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis on all discovered experiments.

        Returns:
            Dictionary containing results for all experiments
        """
        print("🚀 Starting Comprehensive MCTS Analysis")

        experiments = self.discover_experiments()
        all_results = {}

        for exp_key, experiment in experiments.items():
            try:
                results = self.analyze_experiment(experiment)
                if results:
                    all_results[exp_key] = results
                    self.create_performance_visualizations(results)
                    print(f"✅ Completed analysis for {exp_key}")
                else:
                    print(f"❌ Failed to analyze {exp_key}")
            except Exception as e:
                print(f"❌ Error analyzing {exp_key}: {e}")

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, all_results: Dict) -> None:
        """
        Generate a comprehensive summary report.

        Args:
            all_results: Results from all experiments
        """
        report_path = os.path.join(self.output_dir, 'comprehensive_analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE MCTS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments Analyzed: {len(all_results)}\n")
            f.write(f"Analysis Output Directory: {self.output_dir}\n\n")

            for exp_key, results in all_results.items():
                f.write(f"\n--- {exp_key} ---\n")

                if 'data_summary' in results:
                    summary = results['data_summary']
                    f.write(f"Data Samples: Clustered={summary['clustered_samples']}, "
                           f"AllGrid={summary['allgrid_samples']}\n")

                if 'performance_metrics' in results and 'win_rates' in results['performance_metrics']:
                    win_rates = results['performance_metrics']['win_rates']
                    f.write(f"Win Rates: ")
                    for algo, rate in win_rates.items():
                        f.write(f"{algo}={rate:.3f} ")
                    f.write("\n")

                if 'efficiency_analysis' in results and 'overall_efficiency' in results['efficiency_analysis']:
                    eff = results['efficiency_analysis']['overall_efficiency']
                    f.write(f"Efficiency: Clustered={eff['clustered']:.4f}, "
                           f"AllGrid={eff['allgrid']:.4f}, "
                           f"Ratio={eff['efficiency_ratio']:.2f}\n")

                if 'strategy_analysis' in results:
                    strategy = results['strategy_analysis']
                    if 'rotation_agreement_rate' in strategy:
                        f.write(f"Rotation Agreement Rate: {strategy['rotation_agreement_rate']:.3f}\n")
                    if 'velocity_distance_stats' in strategy:
                        dist_stats = strategy['velocity_distance_stats']
                        f.write(f"Avg Velocity Distance: {dist_stats['mean']:.4f} ± {dist_stats['std']:.4f}\n")

        print(f"📊 Summary report generated: {report_path}")


def main():
    """Main function to run the comprehensive analysis."""
    analyzer = ComprehensiveMCTSAnalyzer()
    results = analyzer.run_comprehensive_analysis()

    print(f"\n🎉 Analysis complete! Results saved to: {analyzer.output_dir}")
    print(f"📈 Analyzed {len(results)} experiments with detailed performance comparisons")


if __name__ == "__main__":
    main()