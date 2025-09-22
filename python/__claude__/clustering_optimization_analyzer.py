#!/usr/bin/env python3
"""
Clustering Optimization Analyzer for MCTS Curling AI

This module analyzes the quality and effectiveness of hierarchical clustering used in MCTS,
providing optimization recommendations for clustering parameters and validating clustering decisions.

Features:
- Clustering quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Optimal cluster number determination using multiple methods
- Cluster stability analysis across different game states
- Comparative analysis of different clustering algorithms
- Cluster representative (medoid) quality assessment

Author: Auto-generated analysis tool for ylab-curling project
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ClusteringOptimizationAnalyzer:
    """Analyze and optimize clustering quality for MCTS curling AI."""

    def __init__(self, base_path: str = "../remote_log"):
        """
        Initialize the clustering optimization analyzer.

        Args:
            base_path: Path to the remote_log directory containing experiment results
        """
        self.base_path = base_path
        self.output_dir = "clustering_optimization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set1")
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (14, 10),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def load_clustering_data(self, experiment_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load clustering-related data for analysis.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            Dictionary containing clustering data
        """
        data = {
            'cluster_mappings': [],
            'similarity_matrices': [],
            'shot_scores': []
        }

        # Find all relevant CSV files
        for csv_file in glob.glob(os.path.join(experiment_path, "**", "*.csv"), recursive=True):
            filename = os.path.basename(csv_file)

            try:
                shot_num = self._extract_shot_number(csv_file)

                if "cluster_ids" in filename:
                    df = pd.read_csv(csv_file)
                    df['shot_number'] = shot_num
                    data['cluster_mappings'].append(df)

                elif "similarity" in filename:
                    df = pd.read_csv(csv_file, header=None)  # Similarity matrices usually don't have headers
                    df['shot_number'] = shot_num
                    data['similarity_matrices'].append(df)

                elif "score" in filename and ("clustered" in filename or "allgrid" in filename):
                    df = pd.read_csv(csv_file)
                    df['shot_number'] = shot_num
                    data['shot_scores'].append(df)

            except Exception as e:
                print(f"⚠️ Error loading {csv_file}: {e}")

        # Combine DataFrames
        for key, df_list in data.items():
            if df_list:
                if key != 'similarity_matrices':  # Special handling for similarity matrices
                    data[key] = pd.concat(df_list, ignore_index=True)
                else:
                    data[key] = df_list  # Keep as list for similarity matrices
            else:
                data[key] = pd.DataFrame() if key != 'similarity_matrices' else []

        return data

    def _extract_shot_number(self, file_path: str) -> int:
        """Extract shot number from filename."""
        match = re.search(r"_(\d+)\.csv$", file_path)
        return int(match.group(1)) if match else 0

    def analyze_cluster_quality(self, cluster_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze the quality of existing clustering solutions.

        Args:
            cluster_data: Clustering data from load_clustering_data

        Returns:
            Dictionary containing cluster quality analysis
        """
        results = {}

        if not cluster_data['cluster_mappings'].empty and cluster_data['similarity_matrices']:
            # Analyze each shot's clustering
            shot_quality = []

            for shot_num in cluster_data['cluster_mappings']['shot_number'].unique():
                shot_mappings = cluster_data['cluster_mappings'][
                    cluster_data['cluster_mappings']['shot_number'] == shot_num
                ]

                # Find corresponding similarity matrix
                similarity_matrix = None
                for sim_data in cluster_data['similarity_matrices']:
                    if sim_data['shot_number'] == shot_num:
                        # Remove shot_number column if it exists
                        sim_matrix = sim_data.drop('shot_number', axis=1, errors='ignore')
                        similarity_matrix = sim_matrix.values
                        break

                if similarity_matrix is not None and not shot_mappings.empty:
                    # Convert similarity to distance matrix
                    distance_matrix = np.max(similarity_matrix) - similarity_matrix
                    np.fill_diagonal(distance_matrix, 0)

                    # Get cluster labels
                    state_to_cluster = {}
                    for _, row in shot_mappings.iterrows():
                        state_to_cluster[row['StateId']] = row['ClusterId']

                    # Create labels array
                    n_states = len(distance_matrix)
                    labels = np.array([state_to_cluster.get(i, -1) for i in range(n_states)])

                    # Calculate quality metrics
                    if len(set(labels)) > 1 and -1 not in labels:
                        try:
                            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
                            calinski = calinski_harabasz_score(distance_matrix, labels)
                            davies_bouldin = davies_bouldin_score(distance_matrix, labels)

                            shot_quality.append({
                                'shot_number': shot_num,
                                'silhouette_score': silhouette,
                                'calinski_harabasz_score': calinski,
                                'davies_bouldin_score': davies_bouldin,
                                'num_clusters': len(set(labels)),
                                'num_states': n_states
                            })
                        except Exception as e:
                            print(f"⚠️ Error calculating metrics for shot {shot_num}: {e}")

            results['shot_quality'] = pd.DataFrame(shot_quality)

            # Overall quality statistics
            if shot_quality:
                quality_df = pd.DataFrame(shot_quality)
                results['overall_quality'] = {
                    'avg_silhouette': quality_df['silhouette_score'].mean(),
                    'avg_calinski': quality_df['calinski_harabasz_score'].mean(),
                    'avg_davies_bouldin': quality_df['davies_bouldin_score'].mean(),
                    'std_silhouette': quality_df['silhouette_score'].std(),
                    'avg_clusters': quality_df['num_clusters'].mean()
                }

        return results

    def determine_optimal_clusters(self, cluster_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Determine optimal number of clusters using multiple methods.

        Args:
            cluster_data: Clustering data from load_clustering_data

        Returns:
            Dictionary containing optimal cluster analysis
        """
        results = {}

        if cluster_data['similarity_matrices']:
            optimal_analysis = []

            for sim_data in cluster_data['similarity_matrices'][:5]:  # Analyze first 5 shots
                shot_num = sim_data['shot_number']
                sim_matrix = sim_data.drop('shot_number', axis=1, errors='ignore').values

                # Convert to distance matrix
                distance_matrix = np.max(sim_matrix) - sim_matrix
                np.fill_diagonal(distance_matrix, 0)

                # Test different numbers of clusters
                max_clusters = min(15, len(distance_matrix) // 2)
                cluster_range = range(2, max_clusters + 1)

                silhouette_scores = []
                calinski_scores = []
                davies_bouldin_scores = []
                inertias = []

                for n_clusters in cluster_range:
                    try:
                        # Hierarchical clustering
                        clustering = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric='precomputed',
                            linkage='average'
                        )
                        labels = clustering.fit_predict(distance_matrix)

                        # Calculate metrics
                        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                        cal_score = calinski_harabasz_score(distance_matrix, labels)
                        db_score = davies_bouldin_score(distance_matrix, labels)

                        # Calculate inertia (within-cluster sum of squares)
                        inertia = 0
                        for cluster_id in set(labels):
                            cluster_points = np.where(labels == cluster_id)[0]
                            if len(cluster_points) > 1:
                                cluster_distances = distance_matrix[cluster_points][:, cluster_points]
                                inertia += np.sum(cluster_distances) / (2 * len(cluster_points))

                        silhouette_scores.append(sil_score)
                        calinski_scores.append(cal_score)
                        davies_bouldin_scores.append(db_score)
                        inertias.append(inertia)

                    except Exception as e:
                        print(f"⚠️ Error for {n_clusters} clusters: {e}")
                        silhouette_scores.append(0)
                        calinski_scores.append(0)
                        davies_bouldin_scores.append(float('inf'))
                        inertias.append(float('inf'))

                # Find optimal clusters using different methods
                optimal_methods = {}

                # Silhouette method (maximum)
                if silhouette_scores:
                    optimal_methods['silhouette'] = cluster_range[np.argmax(silhouette_scores)]

                # Calinski-Harabasz method (maximum)
                if calinski_scores:
                    optimal_methods['calinski'] = cluster_range[np.argmax(calinski_scores)]

                # Davies-Bouldin method (minimum)
                if davies_bouldin_scores:
                    optimal_methods['davies_bouldin'] = cluster_range[np.argmin(davies_bouldin_scores)]

                # Elbow method for inertia
                if len(inertias) > 3:
                    try:
                        # Use KneeLocator to find elbow
                        kl = KneeLocator(
                            list(cluster_range), inertias,
                            curve="convex", direction="decreasing"
                        )
                        if kl.elbow:
                            optimal_methods['elbow'] = kl.elbow
                    except:
                        pass

                optimal_analysis.append({
                    'shot_number': shot_num,
                    'cluster_range': list(cluster_range),
                    'silhouette_scores': silhouette_scores,
                    'calinski_scores': calinski_scores,
                    'davies_bouldin_scores': davies_bouldin_scores,
                    'inertias': inertias,
                    'optimal_methods': optimal_methods
                })

            results['optimal_analysis'] = optimal_analysis

            # Consensus optimal cluster number
            if optimal_analysis:
                all_optimal = []
                for analysis in optimal_analysis:
                    all_optimal.extend(analysis['optimal_methods'].values())

                if all_optimal:
                    # Most common optimal cluster number
                    from collections import Counter
                    optimal_counts = Counter(all_optimal)
                    consensus_optimal = optimal_counts.most_common(1)[0][0]
                    results['consensus_optimal'] = consensus_optimal

        return results

    def analyze_cluster_stability(self, cluster_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze cluster stability across different game states.

        Args:
            cluster_data: Clustering data from load_clustering_data

        Returns:
            Dictionary containing stability analysis
        """
        results = {}

        if not cluster_data['cluster_mappings'].empty:
            # Track how cluster assignments change across shots
            stability_metrics = []

            shot_numbers = sorted(cluster_data['cluster_mappings']['shot_number'].unique())

            for i in range(len(shot_numbers) - 1):
                shot1 = shot_numbers[i]
                shot2 = shot_numbers[i + 1]

                mapping1 = cluster_data['cluster_mappings'][
                    cluster_data['cluster_mappings']['shot_number'] == shot1
                ]
                mapping2 = cluster_data['cluster_mappings'][
                    cluster_data['cluster_mappings']['shot_number'] == shot2
                ]

                if not mapping1.empty and not mapping2.empty:
                    # Find common states between shots
                    states1 = set(mapping1['StateId'])
                    states2 = set(mapping2['StateId'])
                    common_states = states1.intersection(states2)

                    if common_states:
                        # Calculate stability metrics
                        stable_assignments = 0
                        total_assignments = len(common_states)

                        for state_id in common_states:
                            cluster1 = mapping1[mapping1['StateId'] == state_id]['ClusterId'].iloc[0]
                            cluster2 = mapping2[mapping2['StateId'] == state_id]['ClusterId'].iloc[0]

                            if cluster1 == cluster2:
                                stable_assignments += 1

                        stability_rate = stable_assignments / total_assignments

                        stability_metrics.append({
                            'shot_pair': f"{shot1}-{shot2}",
                            'stability_rate': stability_rate,
                            'common_states': len(common_states),
                            'stable_assignments': stable_assignments
                        })

            results['stability_metrics'] = pd.DataFrame(stability_metrics)

            # Overall stability statistics
            if stability_metrics:
                stability_df = pd.DataFrame(stability_metrics)
                results['overall_stability'] = {
                    'avg_stability_rate': stability_df['stability_rate'].mean(),
                    'std_stability_rate': stability_df['stability_rate'].std(),
                    'min_stability_rate': stability_df['stability_rate'].min(),
                    'max_stability_rate': stability_df['stability_rate'].max()
                }

        return results

    def compare_clustering_algorithms(self, cluster_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Compare different clustering algorithms on the same data.

        Args:
            cluster_data: Clustering data from load_clustering_data

        Returns:
            Dictionary containing algorithm comparison
        """
        results = {}

        if cluster_data['similarity_matrices']:
            algorithm_comparison = []

            # Test on first few shots
            for sim_data in cluster_data['similarity_matrices'][:3]:
                shot_num = sim_data['shot_number']
                sim_matrix = sim_data.drop('shot_number', axis=1, errors='ignore').values

                # Convert to distance matrix
                distance_matrix = np.max(sim_matrix) - sim_matrix
                np.fill_diagonal(distance_matrix, 0)

                # Test different algorithms
                algorithms = {
                    'hierarchical_average': AgglomerativeClustering(
                        n_clusters=6, metric='precomputed', linkage='average'
                    ),
                    'hierarchical_complete': AgglomerativeClustering(
                        n_clusters=6, metric='precomputed', linkage='complete'
                    ),
                    'hierarchical_single': AgglomerativeClustering(
                        n_clusters=6, metric='precomputed', linkage='single'
                    )
                }

                shot_results = {'shot_number': shot_num}

                for algo_name, algo in algorithms.items():
                    try:
                        labels = algo.fit_predict(distance_matrix)

                        # Calculate quality metrics
                        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                        cal_score = calinski_harabasz_score(distance_matrix, labels)
                        db_score = davies_bouldin_score(distance_matrix, labels)

                        shot_results[f'{algo_name}_silhouette'] = sil_score
                        shot_results[f'{algo_name}_calinski'] = cal_score
                        shot_results[f'{algo_name}_davies_bouldin'] = db_score

                    except Exception as e:
                        print(f"⚠️ Error with {algo_name}: {e}")
                        shot_results[f'{algo_name}_silhouette'] = 0
                        shot_results[f'{algo_name}_calinski'] = 0
                        shot_results[f'{algo_name}_davies_bouldin'] = float('inf')

                algorithm_comparison.append(shot_results)

            results['algorithm_comparison'] = pd.DataFrame(algorithm_comparison)

            # Summary of best algorithms
            if algorithm_comparison:
                comp_df = pd.DataFrame(algorithm_comparison)
                algo_names = ['hierarchical_average', 'hierarchical_complete', 'hierarchical_single']

                best_algorithms = {}
                for metric in ['silhouette', 'calinski', 'davies_bouldin']:
                    metric_cols = [f'{algo}_{metric}' for algo in algo_names]
                    if metric == 'davies_bouldin':
                        # Lower is better for Davies-Bouldin
                        best_algo = comp_df[metric_cols].mean().idxmin().replace(f'_{metric}', '')
                    else:
                        # Higher is better for Silhouette and Calinski-Harabasz
                        best_algo = comp_df[metric_cols].mean().idxmax().replace(f'_{metric}', '')

                    best_algorithms[metric] = best_algo

                results['best_algorithms'] = best_algorithms

        return results

    def create_clustering_visualizations(self, experiment_results: Dict, experiment_name: str) -> None:
        """
        Create comprehensive clustering optimization visualizations.

        Args:
            experiment_results: Results from analyze_experiment
            experiment_name: Name of the experiment for file naming
        """

        # 1. Cluster Quality Over Time
        if 'cluster_quality' in experiment_results and 'shot_quality' in experiment_results['cluster_quality']:
            quality_data = experiment_results['cluster_quality']['shot_quality']

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Silhouette score evolution
            axes[0,0].plot(quality_data['shot_number'], quality_data['silhouette_score'],
                          marker='o', linewidth=2, color='blue')
            axes[0,0].set_title('Silhouette Score Evolution')
            axes[0,0].set_xlabel('Shot Number')
            axes[0,0].set_ylabel('Silhouette Score')
            axes[0,0].grid(True, alpha=0.3)

            # Calinski-Harabasz score evolution
            axes[0,1].plot(quality_data['shot_number'], quality_data['calinski_harabasz_score'],
                          marker='s', linewidth=2, color='green')
            axes[0,1].set_title('Calinski-Harabasz Score Evolution')
            axes[0,1].set_xlabel('Shot Number')
            axes[0,1].set_ylabel('Calinski-Harabasz Score')
            axes[0,1].grid(True, alpha=0.3)

            # Davies-Bouldin score evolution
            axes[1,0].plot(quality_data['shot_number'], quality_data['davies_bouldin_score'],
                          marker='^', linewidth=2, color='red')
            axes[1,0].set_title('Davies-Bouldin Score Evolution (Lower is Better)')
            axes[1,0].set_xlabel('Shot Number')
            axes[1,0].set_ylabel('Davies-Bouldin Score')
            axes[1,0].grid(True, alpha=0.3)

            # Number of clusters evolution
            axes[1,1].plot(quality_data['shot_number'], quality_data['num_clusters'],
                          marker='d', linewidth=2, color='purple')
            axes[1,1].set_title('Number of Clusters Evolution')
            axes[1,1].set_xlabel('Shot Number')
            axes[1,1].set_ylabel('Number of Clusters')
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_quality_evolution.png'))
            plt.close()

        # 2. Optimal Cluster Analysis
        if 'optimal_clusters' in experiment_results and 'optimal_analysis' in experiment_results['optimal_clusters']:
            optimal_data = experiment_results['optimal_clusters']['optimal_analysis']

            if optimal_data:
                # Show optimal cluster analysis for first shot
                first_analysis = optimal_data[0]

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                cluster_range = first_analysis['cluster_range']

                # Silhouette scores
                axes[0,0].plot(cluster_range, first_analysis['silhouette_scores'],
                              marker='o', linewidth=2, color='blue')
                axes[0,0].set_title('Silhouette Score vs Number of Clusters')
                axes[0,0].set_xlabel('Number of Clusters')
                axes[0,0].set_ylabel('Silhouette Score')
                axes[0,0].grid(True, alpha=0.3)

                # Calinski-Harabasz scores
                axes[0,1].plot(cluster_range, first_analysis['calinski_scores'],
                              marker='s', linewidth=2, color='green')
                axes[0,1].set_title('Calinski-Harabasz Score vs Number of Clusters')
                axes[0,1].set_xlabel('Number of Clusters')
                axes[0,1].set_ylabel('Calinski-Harabasz Score')
                axes[0,1].grid(True, alpha=0.3)

                # Davies-Bouldin scores
                axes[1,0].plot(cluster_range, first_analysis['davies_bouldin_scores'],
                              marker='^', linewidth=2, color='red')
                axes[1,0].set_title('Davies-Bouldin Score vs Number of Clusters')
                axes[1,0].set_xlabel('Number of Clusters')
                axes[1,0].set_ylabel('Davies-Bouldin Score')
                axes[1,0].grid(True, alpha=0.3)

                # Inertia (Elbow method)
                axes[1,1].plot(cluster_range, first_analysis['inertias'],
                              marker='d', linewidth=2, color='purple')
                axes[1,1].set_title('Inertia vs Number of Clusters (Elbow Method)')
                axes[1,1].set_xlabel('Number of Clusters')
                axes[1,1].set_ylabel('Inertia')
                axes[1,1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_optimal_clusters.png'))
                plt.close()

        # 3. Algorithm Comparison
        if 'algorithm_comparison' in experiment_results and 'algorithm_comparison' in experiment_results['algorithm_comparison']:
            comp_data = experiment_results['algorithm_comparison']['algorithm_comparison']

            if not comp_data.empty:
                algorithms = ['hierarchical_average', 'hierarchical_complete', 'hierarchical_single']
                metrics = ['silhouette', 'calinski', 'davies_bouldin']

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                for i, metric in enumerate(metrics):
                    metric_cols = [f'{algo}_{metric}' for algo in algorithms]
                    algo_short_names = ['Average', 'Complete', 'Single']

                    # Calculate mean scores for each algorithm
                    means = [comp_data[col].mean() for col in metric_cols]

                    bars = axes[i].bar(algo_short_names, means, alpha=0.7)
                    axes[i].set_title(f'{metric.title()} Score Comparison')
                    axes[i].set_ylabel(f'{metric.title()} Score')
                    axes[i].grid(True, alpha=0.3)

                    # Add value labels on bars
                    for bar, mean in zip(bars, means):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{mean:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_algorithm_comparison.png'))
                plt.close()

        # 4. Cluster Stability Analysis
        if 'cluster_stability' in experiment_results and 'stability_metrics' in experiment_results['cluster_stability']:
            stability_data = experiment_results['cluster_stability']['stability_metrics']

            if not stability_data.empty:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # Stability rate over shot pairs
                axes[0].plot(range(len(stability_data)), stability_data['stability_rate'],
                           marker='o', linewidth=2, color='orange')
                axes[0].set_title('Cluster Stability Between Consecutive Shots')
                axes[0].set_xlabel('Shot Pair Index')
                axes[0].set_ylabel('Stability Rate')
                axes[0].set_xticks(range(len(stability_data)))
                axes[0].set_xticklabels(stability_data['shot_pair'], rotation=45)
                axes[0].grid(True, alpha=0.3)

                # Distribution of stability rates
                axes[1].hist(stability_data['stability_rate'], bins=10, alpha=0.7, color='orange')
                axes[1].set_title('Distribution of Stability Rates')
                axes[1].set_xlabel('Stability Rate')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_stability_analysis.png'))
                plt.close()

    def analyze_experiment(self, experiment_path: str, experiment_name: str) -> Dict:
        """
        Perform complete clustering optimization analysis on a single experiment.

        Args:
            experiment_path: Path to experiment directory
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing all analysis results
        """
        print(f"🔍 Analyzing clustering optimization for: {experiment_name}")

        # Load clustering data
        cluster_data = self.load_clustering_data(experiment_path)

        if not any(not df.empty if isinstance(df, pd.DataFrame) else bool(df) for df in cluster_data.values()):
            print("⚠️ No clustering data found for this experiment")
            return {}

        results = {
            'experiment_name': experiment_name,
            'data_summary': {
                'cluster_mappings_shots': len(cluster_data['cluster_mappings']['shot_number'].unique()) if not cluster_data['cluster_mappings'].empty else 0,
                'similarity_matrices_count': len(cluster_data['similarity_matrices'])
            }
        }

        # Perform analyses
        results['cluster_quality'] = self.analyze_cluster_quality(cluster_data)
        results['optimal_clusters'] = self.determine_optimal_clusters(cluster_data)
        results['cluster_stability'] = self.analyze_cluster_stability(cluster_data)
        results['algorithm_comparison'] = self.compare_clustering_algorithms(cluster_data)

        # Create visualizations
        self.create_clustering_visualizations(results, experiment_name)

        return results

    def run_clustering_analysis(self) -> Dict:
        """
        Run clustering optimization analysis on all discovered experiments.

        Returns:
            Dictionary containing results for all experiments
        """
        print("🚀 Starting Clustering Optimization Analysis")

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
                    cluster_files = [f for f in csv_files if 'cluster' in os.path.basename(f)]

                    if cluster_files:
                        try:
                            results = self.analyze_experiment(experiment_path, dir_name)
                            if results:
                                all_results[dir_name] = results
                                print(f"✅ Completed clustering analysis for {dir_name}")
                        except Exception as e:
                            print(f"❌ Error analyzing {dir_name}: {e}")

        # Generate summary report
        self.generate_clustering_report(all_results)

        return all_results

    def generate_clustering_report(self, all_results: Dict) -> None:
        """
        Generate a comprehensive clustering optimization report.

        Args:
            all_results: Results from all experiments
        """
        report_path = os.path.join(self.output_dir, 'clustering_optimization_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments Analyzed: {len(all_results)}\n")
            f.write(f"Analysis Output Directory: {self.output_dir}\n\n")

            for exp_name, results in all_results.items():
                f.write(f"\n--- {exp_name} ---\n")

                # Overall quality metrics
                if 'cluster_quality' in results and 'overall_quality' in results['cluster_quality']:
                    quality = results['cluster_quality']['overall_quality']
                    f.write(f"Average Silhouette Score: {quality['avg_silhouette']:.3f} ± {quality['std_silhouette']:.3f}\n")
                    f.write(f"Average Calinski-Harabasz Score: {quality['avg_calinski']:.3f}\n")
                    f.write(f"Average Davies-Bouldin Score: {quality['avg_davies_bouldin']:.3f}\n")
                    f.write(f"Average Number of Clusters: {quality['avg_clusters']:.1f}\n")

                # Optimal clusters
                if 'optimal_clusters' in results and 'consensus_optimal' in results['optimal_clusters']:
                    consensus = results['optimal_clusters']['consensus_optimal']
                    f.write(f"Consensus Optimal Clusters: {consensus}\n")

                # Cluster stability
                if 'cluster_stability' in results and 'overall_stability' in results['cluster_stability']:
                    stability = results['cluster_stability']['overall_stability']
                    f.write(f"Average Stability Rate: {stability['avg_stability_rate']:.3f} ± {stability['std_stability_rate']:.3f}\n")

                # Best algorithms
                if 'algorithm_comparison' in results and 'best_algorithms' in results['algorithm_comparison']:
                    best_algos = results['algorithm_comparison']['best_algorithms']
                    f.write("Best Algorithms by Metric:\n")
                    for metric, algo in best_algos.items():
                        f.write(f"  {metric}: {algo}\n")

        print(f"📊 Clustering optimization report generated: {report_path}")


def main():
    """Main function to run the clustering optimization analysis."""
    analyzer = ClusteringOptimizationAnalyzer()
    results = analyzer.run_clustering_analysis()

    print(f"\n🎉 Clustering optimization analysis complete!")
    print(f"🔍 Analyzed {len(results)} experiments with detailed clustering quality assessment")
    print(f"📊 Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()