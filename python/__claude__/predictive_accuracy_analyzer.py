#!/usr/bin/env python3
"""
Predictive Accuracy Analyzer for MCTS Curling AI

This module analyzes the predictive accuracy of MCTS algorithms by comparing
predicted scores/outcomes with actual game results and performance metrics.

Features:
- Prediction vs reality correlation analysis
- Accuracy degradation over game progression
- Confidence interval analysis for predictions
- Ensemble prediction accuracy (combining both algorithms)
- Error pattern identification and root cause analysis

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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class PredictiveAccuracyAnalyzer:
    """Analyze predictive accuracy of MCTS algorithms in curling AI."""

    def __init__(self, base_path: str = "../../remote_log"):
        """
        Initialize the predictive accuracy analyzer.

        Args:
            base_path: Path to the remote_log directory containing experiment results
        """
        self.base_path = base_path
        self.output_dir = "predictive_accuracy_output"
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

    def load_prediction_data(self, experiment_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load prediction and outcome data for accuracy analysis.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            Dictionary containing prediction data
        """
        data = {
            'clustered_predictions': [],
            'allgrid_predictions': [],
            'best_shot_comparisons': [],
            'game_outcomes': []
        }

        # Find all relevant CSV files
        for csv_file in glob.glob(os.path.join(experiment_path, "**", "*.csv"), recursive=True):
            filename = os.path.basename(csv_file)

            try:
                shot_num = self._extract_shot_number(csv_file)

                if "clustered" in filename and "score" in filename:
                    df = pd.read_csv(csv_file)
                    df['shot_number'] = shot_num
                    df['algorithm'] = 'Clustered'
                    data['clustered_predictions'].append(df)

                elif "allgrid" in filename and "score" in filename:
                    df = pd.read_csv(csv_file)
                    df['shot_number'] = shot_num
                    df['algorithm'] = 'AllGrid'
                    data['allgrid_predictions'].append(df)

                elif "comparison" in filename:
                    df = pd.read_csv(csv_file)
                    df['shot_number'] = shot_num
                    data['best_shot_comparisons'].append(df)

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

    def analyze_prediction_accuracy(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze the accuracy of MCTS predictions.

        Args:
            prediction_data: Prediction data from load_prediction_data

        Returns:
            Dictionary containing accuracy analysis
        """
        results = {}

        # Combine prediction data
        if not prediction_data['clustered_predictions'].empty and not prediction_data['allgrid_predictions'].empty:
            all_predictions = pd.concat([
                prediction_data['clustered_predictions'],
                prediction_data['allgrid_predictions']
            ], ignore_index=True)

            # Analyze prediction consistency within each shot
            prediction_consistency = []

            for shot_num in all_predictions['shot_number'].unique():
                shot_data = all_predictions[all_predictions['shot_number'] == shot_num]

                for algorithm in ['Clustered', 'AllGrid']:
                    algo_data = shot_data[shot_data['algorithm'] == algorithm]

                    if not algo_data.empty and len(algo_data) > 1:
                        # Calculate prediction metrics for this shot
                        scores = algo_data['Score']
                        visits = algo_data['Visits']
                        wins = algo_data['Wins']

                        # Prediction uncertainty metrics
                        score_variance = scores.var()
                        score_std = scores.std()
                        score_cv = score_std / scores.mean() if scores.mean() > 0 else 0

                        # Win rate prediction
                        total_visits = visits.sum()
                        total_wins = wins.sum()
                        win_rate = total_wins / total_visits if total_visits > 0 else 0

                        # Best shot confidence (difference between best and second-best)
                        sorted_scores = scores.sort_values(ascending=False)
                        confidence = sorted_scores.iloc[0] - sorted_scores.iloc[1] if len(sorted_scores) > 1 else 0

                        prediction_consistency.append({
                            'shot_number': shot_num,
                            'algorithm': algorithm,
                            'mean_score': scores.mean(),
                            'score_variance': score_variance,
                            'score_std': score_std,
                            'score_cv': score_cv,
                            'win_rate': win_rate,
                            'prediction_confidence': confidence,
                            'num_options': len(algo_data)
                        })

            results['prediction_consistency'] = pd.DataFrame(prediction_consistency)

            # Cross-algorithm prediction agreement
            if not prediction_data['best_shot_comparisons'].empty:
                comparisons = prediction_data['best_shot_comparisons']

                # Separate MCTS and AllGrid best shots
                mcts_shots = comparisons[comparisons['Type'] == 'MCTS']
                allgrid_shots = comparisons[comparisons['Type'] == 'AllGrid']

                if not mcts_shots.empty and not allgrid_shots.empty:
                    merged_best = pd.merge(
                        mcts_shots, allgrid_shots,
                        on='shot_number', suffixes=('_mcts', '_allgrid')
                    )

                    # Calculate agreement metrics
                    velocity_agreement = np.sqrt(
                        (merged_best['Vx_mcts'] - merged_best['Vx_allgrid'])**2 +
                        (merged_best['Vy_mcts'] - merged_best['Vy_allgrid'])**2
                    )

                    rotation_agreement = (merged_best['Rot_mcts'] == merged_best['Rot_allgrid']).astype(int)

                    if 'StateID_mcts' in merged_best.columns and 'StateID_allgrid' in merged_best.columns:
                        state_agreement = (merged_best['StateID_mcts'] == merged_best['StateID_allgrid']).astype(int)
                        results['state_agreement_rate'] = state_agreement.mean()

                    results['algorithm_agreement'] = {
                        'avg_velocity_difference': velocity_agreement.mean(),
                        'std_velocity_difference': velocity_agreement.std(),
                        'rotation_agreement_rate': rotation_agreement.mean(),
                        'agreement_by_shot': pd.DataFrame({
                            'shot_number': merged_best['shot_number'],
                            'velocity_difference': velocity_agreement,
                            'rotation_agreement': rotation_agreement
                        })
                    }

        return results

    def analyze_accuracy_over_time(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze how prediction accuracy changes over the course of the game.

        Args:
            prediction_data: Prediction data from load_prediction_data

        Returns:
            Dictionary containing temporal accuracy analysis
        """
        results = {}

        if not prediction_data['clustered_predictions'].empty and not prediction_data['allgrid_predictions'].empty:
            all_predictions = pd.concat([
                prediction_data['clustered_predictions'],
                prediction_data['allgrid_predictions']
            ], ignore_index=True)

            # Analyze temporal trends
            temporal_trends = []

            shot_numbers = sorted(all_predictions['shot_number'].unique())

            # Calculate rolling metrics
            window_size = 3  # 3-shot rolling window

            for i in range(window_size, len(shot_numbers)):
                window_shots = shot_numbers[i-window_size:i]
                window_data = all_predictions[all_predictions['shot_number'].isin(window_shots)]

                for algorithm in ['Clustered', 'AllGrid']:
                    algo_data = window_data[window_data['algorithm'] == algorithm]

                    if not algo_data.empty:
                        # Calculate window metrics
                        avg_score = algo_data['Score'].mean()
                        score_stability = 1 / (algo_data['Score'].std() + 1e-6)
                        avg_visits = algo_data['Visits'].mean()
                        win_rate = algo_data['Wins'].sum() / algo_data['Visits'].sum()

                        temporal_trends.append({
                            'shot_window_end': shot_numbers[i],
                            'algorithm': algorithm,
                            'avg_score': avg_score,
                            'score_stability': score_stability,
                            'avg_visits': avg_visits,
                            'win_rate': win_rate,
                            'game_phase': 'Early' if shot_numbers[i] <= 4 else 'Mid' if shot_numbers[i] <= 10 else 'Late'
                        })

            results['temporal_trends'] = pd.DataFrame(temporal_trends)

            # Game phase analysis
            if temporal_trends:
                phase_analysis = pd.DataFrame(temporal_trends).groupby(['algorithm', 'game_phase']).agg({
                    'avg_score': ['mean', 'std'],
                    'score_stability': 'mean',
                    'win_rate': 'mean'
                }).round(4)

                results['phase_analysis'] = phase_analysis

        return results

    def analyze_confidence_calibration(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze how well-calibrated the confidence measures are.

        Args:
            prediction_data: Prediction data from load_prediction_data

        Returns:
            Dictionary containing confidence calibration analysis
        """
        results = {}

        if not prediction_data['clustered_predictions'].empty:
            all_predictions = pd.concat([
                prediction_data['clustered_predictions'],
                prediction_data['allgrid_predictions']
            ], ignore_index=True)

            # Analyze relationship between prediction confidence and actual performance
            confidence_analysis = []

            for shot_num in all_predictions['shot_number'].unique():
                shot_data = all_predictions[all_predictions['shot_number'] == shot_num]

                for algorithm in ['Clustered', 'AllGrid']:
                    algo_data = shot_data[shot_data['algorithm'] == algorithm]

                    if not algo_data.empty and len(algo_data) > 1:
                        # Sort by score to find best prediction
                        sorted_data = algo_data.sort_values('Score', ascending=False)
                        best_shot = sorted_data.iloc[0]

                        # Calculate confidence measures
                        score_range = sorted_data['Score'].max() - sorted_data['Score'].min()
                        relative_confidence = (best_shot['Score'] - sorted_data['Score'].mean()) / (sorted_data['Score'].std() + 1e-6)

                        # Visit-based confidence
                        visit_confidence = best_shot['Visits'] / algo_data['Visits'].sum()

                        # Win rate confidence
                        win_rate_confidence = best_shot['Wins'] / (best_shot['Visits'] + 1e-6)

                        confidence_analysis.append({
                            'shot_number': shot_num,
                            'algorithm': algorithm,
                            'best_score': best_shot['Score'],
                            'score_range': score_range,
                            'relative_confidence': relative_confidence,
                            'visit_confidence': visit_confidence,
                            'win_rate_confidence': win_rate_confidence,
                            'num_alternatives': len(algo_data)
                        })

            results['confidence_analysis'] = pd.DataFrame(confidence_analysis)

            # Calibration metrics
            if confidence_analysis:
                conf_df = pd.DataFrame(confidence_analysis)

                # Correlation between different confidence measures
                confidence_correlations = {}
                conf_measures = ['relative_confidence', 'visit_confidence', 'win_rate_confidence']

                for i, measure1 in enumerate(conf_measures):
                    for measure2 in conf_measures[i+1:]:
                        corr, p_val = pearsonr(conf_df[measure1], conf_df[measure2])
                        confidence_correlations[f'{measure1}_vs_{measure2}'] = {
                            'correlation': corr,
                            'p_value': p_val
                        }

                results['confidence_correlations'] = confidence_correlations

        return results

    def analyze_ensemble_accuracy(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze accuracy of ensemble predictions combining both algorithms.

        Args:
            prediction_data: Prediction data from load_prediction_data

        Returns:
            Dictionary containing ensemble analysis
        """
        results = {}

        if (not prediction_data['clustered_predictions'].empty and
            not prediction_data['allgrid_predictions'].empty and
            not prediction_data['best_shot_comparisons'].empty):

            # Create ensemble predictions
            ensemble_analysis = []

            for shot_num in prediction_data['clustered_predictions']['shot_number'].unique():
                clustered_data = prediction_data['clustered_predictions'][
                    prediction_data['clustered_predictions']['shot_number'] == shot_num
                ]
                allgrid_data = prediction_data['allgrid_predictions'][
                    prediction_data['allgrid_predictions']['shot_number'] == shot_num
                ]

                if not clustered_data.empty and not allgrid_data.empty:
                    # Individual algorithm best scores
                    clustered_best = clustered_data.loc[clustered_data['Score'].idxmax()]
                    allgrid_best = allgrid_data.loc[allgrid_data['Score'].idxmax()]

                    # Ensemble strategies
                    ensemble_strategies = {
                        'average_score': (clustered_best['Score'] + allgrid_best['Score']) / 2,
                        'max_score': max(clustered_best['Score'], allgrid_best['Score']),
                        'weighted_by_visits': (
                            (clustered_best['Score'] * clustered_best['Visits'] +
                             allgrid_best['Score'] * allgrid_best['Visits']) /
                            (clustered_best['Visits'] + allgrid_best['Visits'])
                        ),
                        'clustered_score': clustered_best['Score'],
                        'allgrid_score': allgrid_best['Score']
                    }

                    # Agreement-based confidence
                    velocity_diff = np.sqrt(
                        (clustered_best['Vx'] - allgrid_best['Vx'])**2 +
                        (clustered_best['Vy'] - allgrid_best['Vy'])**2
                    )
                    rotation_agreement = clustered_best['Rotation'] == allgrid_best['Rotation']

                    ensemble_analysis.append({
                        'shot_number': shot_num,
                        'velocity_difference': velocity_diff,
                        'rotation_agreement': rotation_agreement,
                        'agreement_confidence': 1 / (1 + velocity_diff),  # Higher when algorithms agree
                        **ensemble_strategies
                    })

            results['ensemble_analysis'] = pd.DataFrame(ensemble_analysis)

            # Ensemble performance metrics
            if ensemble_analysis:
                ensemble_df = pd.DataFrame(ensemble_analysis)

                # Correlation between ensemble strategies and agreement
                strategy_cols = ['average_score', 'max_score', 'weighted_by_visits']
                strategy_correlations = {}

                for strategy in strategy_cols:
                    corr_with_agreement, p_val = pearsonr(
                        ensemble_df[strategy],
                        ensemble_df['agreement_confidence']
                    )
                    strategy_correlations[strategy] = {
                        'correlation_with_agreement': corr_with_agreement,
                        'p_value': p_val
                    }

                results['ensemble_performance'] = strategy_correlations

        return results

    def identify_error_patterns(self, prediction_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Identify patterns in prediction errors and their potential causes.

        Args:
            prediction_data: Prediction data from load_prediction_data

        Returns:
            Dictionary containing error pattern analysis
        """
        results = {}

        if not prediction_data['clustered_predictions'].empty and not prediction_data['allgrid_predictions'].empty:
            all_predictions = pd.concat([
                prediction_data['clustered_predictions'],
                prediction_data['allgrid_predictions']
            ], ignore_index=True)

            # Error pattern analysis
            error_patterns = []

            for shot_num in all_predictions['shot_number'].unique():
                shot_data = all_predictions[all_predictions['shot_number'] == shot_num]

                for algorithm in ['Clustered', 'AllGrid']:
                    algo_data = shot_data[shot_data['algorithm'] == algorithm]

                    if not algo_data.empty and len(algo_data) > 1:
                        # Identify potential error indicators
                        scores = algo_data['Score']
                        visits = algo_data['Visits']

                        # Low confidence indicators
                        score_variance = scores.var()
                        low_visit_shots = (visits < visits.median()).sum()
                        score_outliers = ((scores - scores.mean()).abs() > 2 * scores.std()).sum()

                        # Velocity distribution analysis
                        velocity_magnitude = np.sqrt(algo_data['Vx']**2 + algo_data['Vy']**2)
                        velocity_diversity = velocity_magnitude.std()

                        # Rotation preference bias
                        rotation_bias = abs(algo_data['Rotation'].mean() - 0.5)  # 0.5 would be no bias

                        error_patterns.append({
                            'shot_number': shot_num,
                            'algorithm': algorithm,
                            'score_variance': score_variance,
                            'low_visit_ratio': low_visit_shots / len(algo_data),
                            'outlier_ratio': score_outliers / len(algo_data),
                            'velocity_diversity': velocity_diversity,
                            'rotation_bias': rotation_bias,
                            'prediction_uncertainty': score_variance * velocity_diversity
                        })

            results['error_patterns'] = pd.DataFrame(error_patterns)

            # Identify high-uncertainty predictions
            if error_patterns:
                error_df = pd.DataFrame(error_patterns)

                # Define uncertainty threshold (top 25% of uncertainty scores)
                uncertainty_threshold = error_df['prediction_uncertainty'].quantile(0.75)
                high_uncertainty = error_df[error_df['prediction_uncertainty'] > uncertainty_threshold]

                results['high_uncertainty_analysis'] = {
                    'uncertainty_threshold': uncertainty_threshold,
                    'high_uncertainty_shots': high_uncertainty,
                    'uncertainty_frequency_by_algorithm': high_uncertainty['algorithm'].value_counts().to_dict()
                }

        return results

    def create_accuracy_visualizations(self, experiment_results: Dict, experiment_name: str) -> None:
        """
        Create comprehensive predictive accuracy visualizations.

        Args:
            experiment_results: Results from analyze_experiment
            experiment_name: Name of the experiment for file naming
        """

        # 1. Prediction Consistency Analysis
        if 'prediction_accuracy' in experiment_results and 'prediction_consistency' in experiment_results['prediction_accuracy']:
            consistency_data = experiment_results['prediction_accuracy']['prediction_consistency']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Score variance over time
            for algo in consistency_data['algorithm'].unique():
                algo_data = consistency_data[consistency_data['algorithm'] == algo].sort_values('shot_number')
                axes[0,0].plot(algo_data['shot_number'], algo_data['score_variance'],
                              marker='o', label=algo, linewidth=2)

            axes[0,0].set_title('Score Variance Over Game Progression')
            axes[0,0].set_xlabel('Shot Number')
            axes[0,0].set_ylabel('Score Variance')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Prediction confidence over time
            for algo in consistency_data['algorithm'].unique():
                algo_data = consistency_data[consistency_data['algorithm'] == algo].sort_values('shot_number')
                axes[0,1].plot(algo_data['shot_number'], algo_data['prediction_confidence'],
                              marker='s', label=algo, linewidth=2)

            axes[0,1].set_title('Prediction Confidence Over Game Progression')
            axes[0,1].set_xlabel('Shot Number')
            axes[0,1].set_ylabel('Prediction Confidence')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Win rate predictions
            for algo in consistency_data['algorithm'].unique():
                algo_data = consistency_data[consistency_data['algorithm'] == algo].sort_values('shot_number')
                axes[1,0].plot(algo_data['shot_number'], algo_data['win_rate'],
                              marker='^', label=algo, linewidth=2)

            axes[1,0].set_title('Win Rate Predictions Over Game')
            axes[1,0].set_xlabel('Shot Number')
            axes[1,0].set_ylabel('Win Rate')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

            # Coefficient of variation (stability indicator)
            for algo in consistency_data['algorithm'].unique():
                algo_data = consistency_data[consistency_data['algorithm'] == algo].sort_values('shot_number')
                axes[1,1].plot(algo_data['shot_number'], algo_data['score_cv'],
                              marker='d', label=algo, linewidth=2)

            axes[1,1].set_title('Score Coefficient of Variation (Stability)')
            axes[1,1].set_xlabel('Shot Number')
            axes[1,1].set_ylabel('CV (lower = more stable)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_prediction_consistency.png'))
            plt.close()

        # 2. Algorithm Agreement Analysis
        if 'prediction_accuracy' in experiment_results and 'algorithm_agreement' in experiment_results['prediction_accuracy']:
            agreement = experiment_results['prediction_accuracy']['algorithm_agreement']
            agreement_data = agreement['agreement_by_shot']

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Velocity difference over time
            agreement_data_sorted = agreement_data.sort_values('shot_number')
            axes[0].plot(agreement_data_sorted['shot_number'], agreement_data_sorted['velocity_difference'],
                        marker='o', linewidth=2, color='red')
            axes[0].set_title('Velocity Difference Between Algorithms')
            axes[0].set_xlabel('Shot Number')
            axes[0].set_ylabel('Euclidean Distance in Velocity')
            axes[0].grid(True, alpha=0.3)

            # Rotation agreement over time
            axes[1].plot(agreement_data_sorted['shot_number'], agreement_data_sorted['rotation_agreement'],
                        marker='s', linewidth=2, color='blue')
            axes[1].set_title('Rotation Agreement Between Algorithms')
            axes[1].set_xlabel('Shot Number')
            axes[1].set_ylabel('Agreement (1=agree, 0=disagree)')
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_algorithm_agreement.png'))
            plt.close()

        # 3. Temporal Accuracy Trends
        if 'temporal_accuracy' in experiment_results and 'temporal_trends' in experiment_results['temporal_accuracy']:
            temporal_data = experiment_results['temporal_accuracy']['temporal_trends']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Average score trends
            for algo in temporal_data['algorithm'].unique():
                algo_data = temporal_data[temporal_data['algorithm'] == algo].sort_values('shot_window_end')
                axes[0,0].plot(algo_data['shot_window_end'], algo_data['avg_score'],
                              marker='o', label=algo, linewidth=2)

            axes[0,0].set_title('Average Score Trends (3-shot rolling window)')
            axes[0,0].set_xlabel('Shot Number')
            axes[0,0].set_ylabel('Average Score')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Score stability trends
            for algo in temporal_data['algorithm'].unique():
                algo_data = temporal_data[temporal_data['algorithm'] == algo].sort_values('shot_window_end')
                axes[0,1].plot(algo_data['shot_window_end'], algo_data['score_stability'],
                              marker='s', label=algo, linewidth=2)

            axes[0,1].set_title('Score Stability Trends')
            axes[0,1].set_xlabel('Shot Number')
            axes[0,1].set_ylabel('Stability (1/std)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Game phase comparison
            if 'phase_analysis' in experiment_results['temporal_accuracy']:
                phase_data = experiment_results['temporal_accuracy']['phase_analysis']
                phases = ['Early', 'Mid', 'Late']

                for algo in ['Clustered', 'AllGrid']:
                    if algo in phase_data.index.get_level_values(0):
                        phase_scores = [phase_data.loc[(algo, phase), ('avg_score', 'mean')]
                                      for phase in phases if (algo, phase) in phase_data.index]
                        axes[1,0].plot(phases[:len(phase_scores)], phase_scores,
                                     marker='o', label=algo, linewidth=2)

                axes[1,0].set_title('Average Score by Game Phase')
                axes[1,0].set_xlabel('Game Phase')
                axes[1,0].set_ylabel('Average Score')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)

            # Win rate trends
            for algo in temporal_data['algorithm'].unique():
                algo_data = temporal_data[temporal_data['algorithm'] == algo].sort_values('shot_window_end')
                axes[1,1].plot(algo_data['shot_window_end'], algo_data['win_rate'],
                              marker='^', label=algo, linewidth=2)

            axes[1,1].set_title('Win Rate Trends')
            axes[1,1].set_xlabel('Shot Number')
            axes[1,1].set_ylabel('Win Rate')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_temporal_accuracy.png'))
            plt.close()

        # 4. Error Patterns Visualization
        if 'error_patterns' in experiment_results and 'error_patterns' in experiment_results['error_patterns']:
            error_data = experiment_results['error_patterns']['error_patterns']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Prediction uncertainty over time
            for algo in error_data['algorithm'].unique():
                algo_data = error_data[error_data['algorithm'] == algo].sort_values('shot_number')
                axes[0,0].plot(algo_data['shot_number'], algo_data['prediction_uncertainty'],
                              marker='o', label=algo, linewidth=2)

            axes[0,0].set_title('Prediction Uncertainty Over Game')
            axes[0,0].set_xlabel('Shot Number')
            axes[0,0].set_ylabel('Uncertainty Score')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

            # Velocity diversity
            for algo in error_data['algorithm'].unique():
                algo_data = error_data[error_data['algorithm'] == algo].sort_values('shot_number')
                axes[0,1].plot(algo_data['shot_number'], algo_data['velocity_diversity'],
                              marker='s', label=algo, linewidth=2)

            axes[0,1].set_title('Velocity Diversity in Predictions')
            axes[0,1].set_xlabel('Shot Number')
            axes[0,1].set_ylabel('Velocity Std Dev')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Rotation bias
            for algo in error_data['algorithm'].unique():
                algo_data = error_data[error_data['algorithm'] == algo].sort_values('shot_number')
                axes[1,0].plot(algo_data['shot_number'], algo_data['rotation_bias'],
                              marker='^', label=algo, linewidth=2)

            axes[1,0].set_title('Rotation Bias in Predictions')
            axes[1,0].set_xlabel('Shot Number')
            axes[1,0].set_ylabel('Bias (0.5 = no bias)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

            # Low visit ratio
            for algo in error_data['algorithm'].unique():
                algo_data = error_data[error_data['algorithm'] == algo].sort_values('shot_number')
                axes[1,1].plot(algo_data['shot_number'], algo_data['low_visit_ratio'],
                              marker='d', label=algo, linewidth=2)

            axes[1,1].set_title('Low Visit Ratio (Insufficient Exploration)')
            axes[1,1].set_xlabel('Shot Number')
            axes[1,1].set_ylabel('Ratio of Low-Visit Shots')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_error_patterns.png'))
            plt.close()

    def analyze_experiment(self, experiment_path: str, experiment_name: str) -> Dict:
        """
        Perform complete predictive accuracy analysis on a single experiment.

        Args:
            experiment_path: Path to experiment directory
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing all analysis results
        """
        print(f"🎯 Analyzing predictive accuracy for: {experiment_name}")

        # Load prediction data
        prediction_data = self.load_prediction_data(experiment_path)

        if not any(not df.empty for df in prediction_data.values()):
            print("⚠️ No prediction data found for this experiment")
            return {}

        results = {
            'experiment_name': experiment_name,
            'data_summary': {
                'clustered_predictions': len(prediction_data['clustered_predictions']),
                'allgrid_predictions': len(prediction_data['allgrid_predictions']),
                'shot_comparisons': len(prediction_data['best_shot_comparisons'])
            }
        }

        # Perform analyses
        results['prediction_accuracy'] = self.analyze_prediction_accuracy(prediction_data)
        results['temporal_accuracy'] = self.analyze_accuracy_over_time(prediction_data)
        results['confidence_calibration'] = self.analyze_confidence_calibration(prediction_data)
        results['ensemble_accuracy'] = self.analyze_ensemble_accuracy(prediction_data)
        results['error_patterns'] = self.identify_error_patterns(prediction_data)

        # Create visualizations
        self.create_accuracy_visualizations(results, experiment_name)

        return results

    def run_accuracy_analysis(self) -> Dict:
        """
        Run predictive accuracy analysis on all discovered experiments.

        Returns:
            Dictionary containing results for all experiments
        """
        print("🚀 Starting Predictive Accuracy Analysis")

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
                                print(f"✅ Completed accuracy analysis for {dir_name}")
                        except Exception as e:
                            print(f"❌ Error analyzing {dir_name}: {e}")

        # Generate summary report
        self.generate_accuracy_report(all_results)

        return all_results

    def generate_accuracy_report(self, all_results: Dict) -> None:
        """
        Generate a comprehensive predictive accuracy report.

        Args:
            all_results: Results from all experiments
        """
        report_path = os.path.join(self.output_dir, 'predictive_accuracy_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PREDICTIVE ACCURACY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments Analyzed: {len(all_results)}\n")
            f.write(f"Analysis Output Directory: {self.output_dir}\n\n")

            for exp_name, results in all_results.items():
                f.write(f"\n--- {exp_name} ---\n")

                # Algorithm agreement metrics
                if 'prediction_accuracy' in results and 'algorithm_agreement' in results['prediction_accuracy']:
                    agreement = results['prediction_accuracy']['algorithm_agreement']
                    f.write(f"Average Velocity Difference: {agreement['avg_velocity_difference']:.4f}\n")
                    f.write(f"Rotation Agreement Rate: {agreement['rotation_agreement_rate']:.3f}\n")

                # Confidence calibration
                if 'confidence_calibration' in results and 'confidence_correlations' in results['confidence_calibration']:
                    f.write("Confidence Measure Correlations:\n")
                    correlations = results['confidence_calibration']['confidence_correlations']
                    for measure_pair, stats in correlations.items():
                        f.write(f"  {measure_pair}: r={stats['correlation']:.3f}, p={stats['p_value']:.3f}\n")

                # Error patterns
                if 'error_patterns' in results and 'high_uncertainty_analysis' in results['error_patterns']:
                    uncertainty = results['error_patterns']['high_uncertainty_analysis']
                    f.write(f"High Uncertainty Threshold: {uncertainty['uncertainty_threshold']:.4f}\n")
                    freq = uncertainty['uncertainty_frequency_by_algorithm']
                    for algo, count in freq.items():
                        f.write(f"  {algo} High Uncertainty Shots: {count}\n")

        print(f"📊 Predictive accuracy report generated: {report_path}")


def main():
    """Main function to run the predictive accuracy analysis."""
    analyzer = PredictiveAccuracyAnalyzer()
    results = analyzer.run_accuracy_analysis()

    print(f"\n🎉 Predictive accuracy analysis complete!")
    print(f"🎯 Analyzed {len(results)} experiments with detailed accuracy assessment")
    print(f"📊 Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()