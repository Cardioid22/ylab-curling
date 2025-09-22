#!/usr/bin/env python3
"""
Strategy Evolution Tracker for MCTS Curling AI

This module analyzes how curling strategies evolve throughout the game, identifying
key strategic turning points, decision patterns, and critical moments that influence game outcomes.

Features:
- Game progression analysis with strategy evolution tracking
- Critical turn identification (where strategies diverge significantly)
- Strategic pattern recognition and clustering
- Turning point detection and impact analysis
- Team-specific strategy characterization

Author: Auto-generated analysis tool for ylab-curling project
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class StrategyEvolutionTracker:
    """Analyze strategy evolution patterns in MCTS curling games."""

    def __init__(self, base_path: str = "../remote_log"):
        """
        Initialize the strategy evolution tracker.

        Args:
            base_path: Path to the remote_log directory containing experiment results
        """
        self.base_path = base_path
        self.output_dir = "strategy_evolution_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (14, 10),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def load_game_data(self, experiment_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load and organize game data for strategy analysis.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            Dictionary containing organized game data
        """
        data = {
            'clustered_scores': [],
            'allgrid_scores': [],
            'best_shots': [],
            'cluster_mappings': []
        }

        # Find all relevant CSV files
        for csv_file in glob.glob(os.path.join(experiment_path, "**", "*.csv"), recursive=True):
            filename = os.path.basename(csv_file)

            try:
                df = pd.read_csv(csv_file)
                shot_num = self._extract_shot_number(csv_file)

                if "clustered" in filename and "score" in filename:
                    df['shot_number'] = shot_num
                    df['algorithm'] = 'Clustered'
                    data['clustered_scores'].append(df)

                elif "allgrid" in filename and "score" in filename:
                    df['shot_number'] = shot_num
                    df['algorithm'] = 'AllGrid'
                    data['allgrid_scores'].append(df)

                elif "comparison" in filename:
                    df['shot_number'] = shot_num
                    data['best_shots'].append(df)

                elif "cluster_ids" in filename:
                    df['shot_number'] = shot_num
                    data['cluster_mappings'].append(df)

            except Exception as e:
                print(f"⚠️ Error loading {csv_file}: {e}")

        # Combine DataFrames
        for key, df_list in data.items():
            if df_list:
                data[key] = pd.concat(df_list, ignore_index=True)
            else:
                data[key] = pd.DataFrame()

        return data

    def _extract_shot_number(self, file_path: str) -> int:
        """Extract shot number from filename."""
        match = re.search(r"_(\d+)\.csv$", file_path)
        return int(match.group(1)) if match else 0

    def analyze_strategy_patterns(self, game_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze strategic patterns throughout the game.

        Args:
            game_data: Game data from load_game_data

        Returns:
            Dictionary containing strategy pattern analysis
        """
        results = {}

        # Combine MCTS data for analysis
        if not game_data['clustered_scores'].empty and not game_data['allgrid_scores'].empty:
            all_mcts = pd.concat([game_data['clustered_scores'], game_data['allgrid_scores']],
                               ignore_index=True)

            # Strategy features for each shot
            strategy_features = all_mcts.groupby(['shot_number', 'algorithm']).agg({
                'Score': ['mean', 'std', 'max', 'min'],
                'Visits': ['mean', 'sum'],
                'Wins': ['mean', 'sum'],
                'Vx': ['mean', 'std'],
                'Vy': ['mean', 'std'],
                'Rotation': lambda x: (x == 0).mean()  # CCW ratio
            }).reset_index()

            # Flatten column names
            strategy_features.columns = ['_'.join(col).strip() if col[1] else col[0]
                                       for col in strategy_features.columns.values]

            results['strategy_features'] = strategy_features

            # Calculate strategy diversity (entropy of shot selections)
            strategy_diversity = []
            for shot_num in all_mcts['shot_number'].unique():
                shot_data = all_mcts[all_mcts['shot_number'] == shot_num]

                for algo in ['Clustered', 'AllGrid']:
                    algo_data = shot_data[shot_data['algorithm'] == algo]
                    if not algo_data.empty:
                        # Discretize velocity space and calculate entropy
                        vx_bins = pd.cut(algo_data['Vx'], bins=10, labels=False)
                        vy_bins = pd.cut(algo_data['Vy'], bins=10, labels=False)

                        # Create velocity combinations
                        velocity_combos = vx_bins * 10 + vy_bins
                        value_counts = pd.Series(velocity_combos).value_counts(normalize=True)
                        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))

                        strategy_diversity.append({
                            'shot_number': shot_num,
                            'algorithm': algo,
                            'diversity_entropy': entropy,
                            'unique_strategies': len(value_counts)
                        })

            results['strategy_diversity'] = pd.DataFrame(strategy_diversity)

        return results

    def identify_critical_turns(self, game_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Identify critical turns where strategies diverge significantly.

        Args:
            game_data: Game data from load_game_data

        Returns:
            Dictionary containing critical turn analysis
        """
        results = {}

        if not game_data['best_shots'].empty:
            best_shots = game_data['best_shots']

            # Separate MCTS and AllGrid shots
            mcts_shots = best_shots[best_shots['Type'] == 'MCTS'].sort_values('shot_number')
            allgrid_shots = best_shots[best_shots['Type'] == 'AllGrid'].sort_values('shot_number')

            if not mcts_shots.empty and not allgrid_shots.empty:
                # Merge for comparison
                merged = pd.merge(mcts_shots, allgrid_shots, on='shot_number', suffixes=('_mcts', '_allgrid'))

                # Calculate strategic divergence metrics
                merged['velocity_divergence'] = np.sqrt(
                    (merged['Vx_mcts'] - merged['Vx_allgrid'])**2 +
                    (merged['Vy_mcts'] - merged['Vy_allgrid'])**2
                )

                merged['rotation_divergence'] = (merged['Rot_mcts'] != merged['Rot_allgrid']).astype(int)

                # State divergence if available
                if 'StateID_mcts' in merged.columns and 'StateID_allgrid' in merged.columns:
                    merged['state_divergence'] = (merged['StateID_mcts'] != merged['StateID_allgrid']).astype(int)

                # Identify peaks in divergence
                velocity_peaks, _ = find_peaks(merged['velocity_divergence'],
                                             height=merged['velocity_divergence'].std())

                critical_turns = merged.iloc[velocity_peaks]
                results['critical_turns'] = critical_turns

                # Calculate turn criticality score
                merged['criticality_score'] = (
                    merged['velocity_divergence'] / merged['velocity_divergence'].max() +
                    merged['rotation_divergence']
                )

                results['turn_analysis'] = merged[['shot_number', 'velocity_divergence',
                                                'rotation_divergence', 'criticality_score']]

                # Game phase analysis (early, mid, late game)
                max_shot = merged['shot_number'].max()
                merged['game_phase'] = pd.cut(merged['shot_number'],
                                            bins=[0, max_shot/3, 2*max_shot/3, max_shot],
                                            labels=['Early', 'Mid', 'Late'])

                phase_divergence = merged.groupby('game_phase').agg({
                    'velocity_divergence': ['mean', 'std', 'max'],
                    'rotation_divergence': 'mean',
                    'criticality_score': 'mean'
                })

                results['phase_analysis'] = phase_divergence

        return results

    def analyze_strategic_transitions(self, game_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze how strategies transition between different types.

        Args:
            game_data: Game data from load_game_data

        Returns:
            Dictionary containing strategic transition analysis
        """
        results = {}

        if not game_data['clustered_scores'].empty:
            all_mcts = pd.concat([game_data['clustered_scores'], game_data['allgrid_scores']],
                               ignore_index=True)

            # Define strategy types based on shot characteristics
            def categorize_strategy(row):
                vx, vy, rotation = row['Vx'], row['Vy'], row['Rotation']

                # Aggressive vs Conservative (based on velocity magnitude)
                velocity_mag = np.sqrt(vx**2 + vy**2)

                if velocity_mag < 2.3:
                    return 'Conservative'
                elif velocity_mag > 2.5:
                    return 'Aggressive'
                else:
                    return 'Moderate'

            # Categorize each shot
            all_mcts['strategy_type'] = all_mcts.apply(categorize_strategy, axis=1)

            # Analyze transitions for each algorithm
            for algo in ['Clustered', 'AllGrid']:
                algo_data = all_mcts[all_mcts['algorithm'] == algo].sort_values('shot_number')

                if len(algo_data) > 1:
                    # Get best shots for each turn (highest score)
                    best_shots = algo_data.loc[algo_data.groupby('shot_number')['Score'].idxmax()]

                    # Calculate transition matrix
                    strategy_types = ['Conservative', 'Moderate', 'Aggressive']
                    transition_matrix = pd.DataFrame(0, index=strategy_types, columns=strategy_types)

                    for i in range(len(best_shots) - 1):
                        current = best_shots.iloc[i]['strategy_type']
                        next_strategy = best_shots.iloc[i+1]['strategy_type']
                        transition_matrix.loc[current, next_strategy] += 1

                    # Normalize to probabilities
                    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

                    results[f'{algo.lower()}_transitions'] = transition_matrix

                    # Strategy persistence (how often strategies stay the same)
                    persistence = np.diag(transition_matrix).mean()
                    results[f'{algo.lower()}_persistence'] = persistence

        return results

    def detect_turning_points(self, game_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Detect significant turning points in game strategy.

        Args:
            game_data: Game data from load_game_data

        Returns:
            Dictionary containing turning point analysis
        """
        results = {}

        if not game_data['clustered_scores'].empty and not game_data['allgrid_scores'].empty:
            # Analyze score trends
            score_trends = {}

            for algo in ['Clustered', 'AllGrid']:
                if algo == 'Clustered':
                    data = game_data['clustered_scores']
                else:
                    data = game_data['allgrid_scores']

                # Get average score per shot
                avg_scores = data.groupby('shot_number')['Score'].mean()

                # Detect change points using moving average
                window = 3
                if len(avg_scores) >= window:
                    moving_avg = avg_scores.rolling(window=window, center=True).mean()

                    # Calculate slope changes
                    slopes = np.diff(moving_avg.dropna())
                    slope_changes = np.diff(slopes)

                    # Find significant slope changes (turning points)
                    threshold = np.std(slope_changes) * 1.5
                    turning_points = np.where(np.abs(slope_changes) > threshold)[0]

                    score_trends[algo] = {
                        'scores': avg_scores,
                        'moving_avg': moving_avg,
                        'turning_points': turning_points + window//2 + 1,  # Adjust for window offset
                        'slope_changes': slope_changes
                    }

            results['score_trends'] = score_trends

            # Strategic momentum analysis
            momentum_data = []
            all_data = pd.concat([game_data['clustered_scores'], game_data['allgrid_scores']])

            for shot_num in sorted(all_data['shot_number'].unique()):
                shot_data = all_data[all_data['shot_number'] == shot_num]

                for algo in ['Clustered', 'AllGrid']:
                    algo_data = shot_data[shot_data['algorithm'] == algo]
                    if not algo_data.empty:
                        # Calculate momentum metrics
                        avg_score = algo_data['Score'].mean()
                        score_variance = algo_data['Score'].var()
                        top_score = algo_data['Score'].max()

                        momentum_data.append({
                            'shot_number': shot_num,
                            'algorithm': algo,
                            'avg_score': avg_score,
                            'score_variance': score_variance,
                            'top_score': top_score,
                            'momentum': avg_score * (1 - score_variance/avg_score if avg_score > 0 else 0)
                        })

            results['momentum_analysis'] = pd.DataFrame(momentum_data)

        return results

    def create_evolution_visualizations(self, experiment_results: Dict, experiment_name: str) -> None:
        """
        Create comprehensive strategy evolution visualizations.

        Args:
            experiment_results: Results from analyze_experiment
            experiment_name: Name of the experiment for file naming
        """

        # 1. Strategy Evolution Over Time
        if 'strategy_patterns' in experiment_results:
            patterns = experiment_results['strategy_patterns']

            if 'strategy_features' in patterns:
                features = patterns['strategy_features']

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # Score evolution
                for algo in features['algorithm_'].unique():
                    algo_data = features[features['algorithm_'] == algo]
                    axes[0,0].plot(algo_data['shot_number_'], algo_data['Score_mean'],
                                 marker='o', label=algo, linewidth=2)
                axes[0,0].set_title('Score Evolution')
                axes[0,0].set_xlabel('Shot Number')
                axes[0,0].set_ylabel('Average Score')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)

                # Velocity patterns
                for algo in features['algorithm_'].unique():
                    algo_data = features[features['algorithm_'] == algo]
                    axes[0,1].plot(algo_data['shot_number_'], algo_data['Vx_mean'],
                                 marker='s', label=f'{algo} Vx', linewidth=2)
                    axes[0,2].plot(algo_data['shot_number_'], algo_data['Vy_mean'],
                                 marker='^', label=f'{algo} Vy', linewidth=2)

                axes[0,1].set_title('Velocity X Evolution')
                axes[0,1].set_xlabel('Shot Number')
                axes[0,1].set_ylabel('Average Vx')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)

                axes[0,2].set_title('Velocity Y Evolution')
                axes[0,2].set_xlabel('Shot Number')
                axes[0,2].set_ylabel('Average Vy')
                axes[0,2].legend()
                axes[0,2].grid(True, alpha=0.3)

                # Strategy diversity
                if 'strategy_diversity' in patterns:
                    diversity = patterns['strategy_diversity']
                    for algo in diversity['algorithm'].unique():
                        algo_div = diversity[diversity['algorithm'] == algo]
                        axes[1,0].plot(algo_div['shot_number'], algo_div['diversity_entropy'],
                                     marker='o', label=algo, linewidth=2)
                    axes[1,0].set_title('Strategy Diversity (Entropy)')
                    axes[1,0].set_xlabel('Shot Number')
                    axes[1,0].set_ylabel('Entropy')
                    axes[1,0].legend()
                    axes[1,0].grid(True, alpha=0.3)

                # Visits evolution
                for algo in features['algorithm_'].unique():
                    algo_data = features[features['algorithm_'] == algo]
                    axes[1,1].plot(algo_data['shot_number_'], algo_data['Visits_sum'],
                                 marker='d', label=algo, linewidth=2)
                axes[1,1].set_title('Total Visits Evolution')
                axes[1,1].set_xlabel('Shot Number')
                axes[1,1].set_ylabel('Total Visits')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)

                # Rotation preference
                for algo in features['algorithm_'].unique():
                    algo_data = features[features['algorithm_'] == algo]
                    axes[1,2].plot(algo_data['shot_number_'], algo_data['Rotation_<lambda>'],
                                 marker='h', label=f'{algo} CCW Rate', linewidth=2)
                axes[1,2].set_title('Rotation Preference (CCW Rate)')
                axes[1,2].set_xlabel('Shot Number')
                axes[1,2].set_ylabel('CCW Rate')
                axes[1,2].legend()
                axes[1,2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_strategy_evolution.png'))
                plt.close()

        # 2. Critical Turns and Divergence Analysis
        if 'critical_turns' in experiment_results:
            critical = experiment_results['critical_turns']

            if 'turn_analysis' in critical:
                turn_data = critical['turn_analysis']

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # Velocity divergence
                axes[0,0].plot(turn_data['shot_number'], turn_data['velocity_divergence'],
                             marker='o', color='red', linewidth=2)
                axes[0,0].set_title('Velocity Divergence Between Algorithms')
                axes[0,0].set_xlabel('Shot Number')
                axes[0,0].set_ylabel('Euclidean Distance')
                axes[0,0].grid(True, alpha=0.3)

                # Rotation divergence
                axes[0,1].bar(turn_data['shot_number'], turn_data['rotation_divergence'],
                            alpha=0.7, color='blue')
                axes[0,1].set_title('Rotation Divergence')
                axes[0,1].set_xlabel('Shot Number')
                axes[0,1].set_ylabel('Different Rotation (0/1)')
                axes[0,1].grid(True, alpha=0.3)

                # Criticality score
                axes[1,0].plot(turn_data['shot_number'], turn_data['criticality_score'],
                             marker='s', color='green', linewidth=2)
                axes[1,0].set_title('Turn Criticality Score')
                axes[1,0].set_xlabel('Shot Number')
                axes[1,0].set_ylabel('Criticality Score')
                axes[1,0].grid(True, alpha=0.3)

                # Phase analysis
                if 'phase_analysis' in critical:
                    phase_data = critical['phase_analysis']
                    phases = ['Early', 'Mid', 'Late']
                    velocity_means = [phase_data.loc[phase, ('velocity_divergence', 'mean')]
                                    for phase in phases if phase in phase_data.index]

                    if velocity_means:
                        axes[1,1].bar(phases[:len(velocity_means)], velocity_means,
                                    alpha=0.7, color=['orange', 'purple', 'brown'][:len(velocity_means)])
                        axes[1,1].set_title('Average Divergence by Game Phase')
                        axes[1,1].set_xlabel('Game Phase')
                        axes[1,1].set_ylabel('Average Velocity Divergence')
                        axes[1,1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_critical_turns.png'))
                plt.close()

        # 3. Strategic Transitions Heatmap
        if 'strategic_transitions' in experiment_results:
            transitions = experiment_results['strategic_transitions']

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Clustered transitions
            if 'clustered_transitions' in transitions:
                sns.heatmap(transitions['clustered_transitions'], annot=True, fmt='.2f',
                          cmap='Blues', ax=axes[0])
                axes[0].set_title('Clustered MCTS Strategy Transitions')
                axes[0].set_xlabel('Next Strategy')
                axes[0].set_ylabel('Current Strategy')

            # AllGrid transitions
            if 'allgrid_transitions' in transitions:
                sns.heatmap(transitions['allgrid_transitions'], annot=True, fmt='.2f',
                          cmap='Reds', ax=axes[1])
                axes[1].set_title('AllGrid MCTS Strategy Transitions')
                axes[1].set_xlabel('Next Strategy')
                axes[1].set_ylabel('Current Strategy')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_strategy_transitions.png'))
            plt.close()

    def analyze_experiment(self, experiment_path: str, experiment_name: str) -> Dict:
        """
        Perform complete strategy evolution analysis on a single experiment.

        Args:
            experiment_path: Path to experiment directory
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing all analysis results
        """
        print(f"🎯 Analyzing strategy evolution for: {experiment_name}")

        # Load game data
        game_data = self.load_game_data(experiment_path)

        if not any(not df.empty for df in game_data.values()):
            print("⚠️ No data found for this experiment")
            return {}

        results = {
            'experiment_name': experiment_name,
            'data_summary': {
                'clustered_samples': len(game_data['clustered_scores']),
                'allgrid_samples': len(game_data['allgrid_scores']),
                'shots_compared': len(game_data['best_shots'])
            }
        }

        # Perform analyses
        results['strategy_patterns'] = self.analyze_strategy_patterns(game_data)
        results['critical_turns'] = self.identify_critical_turns(game_data)
        results['strategic_transitions'] = self.analyze_strategic_transitions(game_data)
        results['turning_points'] = self.detect_turning_points(game_data)

        # Create visualizations
        self.create_evolution_visualizations(results, experiment_name)

        return results

    def run_evolution_analysis(self) -> Dict:
        """
        Run strategy evolution analysis on all discovered experiments.

        Returns:
            Dictionary containing results for all experiments
        """
        print("🚀 Starting Strategy Evolution Analysis")

        all_results = {}

        # Discover experiment directories
        if not os.path.exists(self.base_path):
            print(f"❌ Base path {self.base_path} does not exist!")
            return all_results

        for root, dirs, files in os.walk(self.base_path):
            for dir_name in dirs:
                if any(term in dir_name.lower() for term in ['grid', 'iter', 'mcts']):
                    experiment_path = os.path.join(root, dir_name)

                    # Check if directory contains relevant CSV files
                    csv_files = glob.glob(os.path.join(experiment_path, "**", "*.csv"), recursive=True)
                    if csv_files:
                        try:
                            results = self.analyze_experiment(experiment_path, dir_name)
                            if results:
                                all_results[dir_name] = results
                                print(f"✅ Completed analysis for {dir_name}")
                        except Exception as e:
                            print(f"❌ Error analyzing {dir_name}: {e}")

        # Generate summary report
        self.generate_evolution_report(all_results)

        return all_results

    def generate_evolution_report(self, all_results: Dict) -> None:
        """
        Generate a comprehensive strategy evolution report.

        Args:
            all_results: Results from all experiments
        """
        report_path = os.path.join(self.output_dir, 'strategy_evolution_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STRATEGY EVOLUTION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments Analyzed: {len(all_results)}\n")
            f.write(f"Analysis Output Directory: {self.output_dir}\n\n")

            for exp_name, results in all_results.items():
                f.write(f"\n--- {exp_name} ---\n")

                if 'data_summary' in results:
                    summary = results['data_summary']
                    f.write(f"Data: {summary['clustered_samples']} clustered, "
                           f"{summary['allgrid_samples']} allgrid samples\n")

                # Critical turns analysis
                if 'critical_turns' in results and 'critical_turns' in results['critical_turns']:
                    critical_turns = results['critical_turns']['critical_turns']
                    f.write(f"Critical Turns Identified: {len(critical_turns)}\n")
                    if not critical_turns.empty:
                        avg_divergence = critical_turns['velocity_divergence'].mean()
                        f.write(f"Average Velocity Divergence at Critical Turns: {avg_divergence:.4f}\n")

                # Strategic persistence
                if 'strategic_transitions' in results:
                    transitions = results['strategic_transitions']
                    if 'clustered_persistence' in transitions:
                        f.write(f"Clustered Strategy Persistence: {transitions['clustered_persistence']:.3f}\n")
                    if 'allgrid_persistence' in transitions:
                        f.write(f"AllGrid Strategy Persistence: {transitions['allgrid_persistence']:.3f}\n")

                # Turning points
                if 'turning_points' in results and 'score_trends' in results['turning_points']:
                    trends = results['turning_points']['score_trends']
                    for algo, trend_data in trends.items():
                        turning_points = trend_data['turning_points']
                        f.write(f"{algo} Score Turning Points: {len(turning_points)} detected\n")

        print(f"📊 Strategy evolution report generated: {report_path}")


def main():
    """Main function to run the strategy evolution analysis."""
    tracker = StrategyEvolutionTracker()
    results = tracker.run_evolution_analysis()

    print(f"\n🎉 Strategy evolution analysis complete!")
    print(f"📈 Analyzed {len(results)} experiments with detailed evolution tracking")
    print(f"📊 Results saved to: {tracker.output_dir}")


if __name__ == "__main__":
    main()