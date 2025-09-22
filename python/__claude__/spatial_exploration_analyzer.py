#!/usr/bin/env python3
"""
Spatial Exploration Analyzer for MCTS Curling AI

This module analyzes spatial exploration patterns in curling AI decision-making,
identifying hotspots, exploration density, and strategic area preferences on the ice.

Features:
- Spatial density analysis and heatmap generation
- Strategic hotspot identification
- Exploration efficiency metrics
- Zone-based strategic analysis (house, guard zone, etc.)
- Comparative spatial pattern analysis between algorithms

Author: Auto-generated analysis tool for ylab-curling project
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle, Ellipse
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SpatialExplorationAnalyzer:
    """Analyze spatial exploration patterns in MCTS curling AI."""

    def __init__(self, base_path: str = "../remote_log"):
        """
        Initialize the spatial exploration analyzer.

        Args:
            base_path: Path to the remote_log directory containing experiment results
        """
        self.base_path = base_path
        self.output_dir = "spatial_exploration_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Curling ice dimensions and zones
        self.house_radius = 1.829
        self.house_center_x = 0.0
        self.house_center_y = 38.405
        self.area_max_x = 2.375
        self.area_max_y = 40.234

        # Define strategic zones
        self.zones = {
            'house': {'center': (0, 38.405), 'radius': 1.829},
            'button': {'center': (0, 38.405), 'radius': 0.15},
            'four_foot': {'center': (0, 38.405), 'radius': 0.61},
            'eight_foot': {'center': (0, 38.405), 'radius': 1.22},
            'twelve_foot': {'center': (0, 38.405), 'radius': 1.829},
            'guard_zone': {'y_range': (36.0, 37.5), 'x_range': (-2.375, 2.375)},
            'back_line': {'y_range': (39.3, 40.234), 'x_range': (-2.375, 2.375)}
        }

        # Setup plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (14, 10),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

    def load_spatial_data(self, experiment_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load spatial data for analysis.

        Args:
            experiment_path: Path to experiment directory

        Returns:
            Dictionary containing spatial data
        """
        data = {
            'clustered_shots': [],
            'allgrid_shots': [],
            'best_shots': []
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
                    data['clustered_shots'].append(df)

                elif "allgrid" in filename and "score" in filename:
                    df['shot_number'] = shot_num
                    df['algorithm'] = 'AllGrid'
                    data['allgrid_shots'].append(df)

                elif "comparison" in filename:
                    df['shot_number'] = shot_num
                    data['best_shots'].append(df)

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

    def calculate_spatial_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate spatial exploration metrics.

        Args:
            data: Spatial data from load_spatial_data

        Returns:
            Dictionary containing spatial metrics
        """
        results = {}

        # Combine all shot data
        if not data['clustered_shots'].empty and not data['allgrid_shots'].empty:
            all_shots = pd.concat([data['clustered_shots'], data['allgrid_shots']], ignore_index=True)

            # Convert velocity to position estimates (simplified physics model)
            all_shots['estimated_x'] = all_shots['Vx'] * 0.5  # Simplified landing position
            all_shots['estimated_y'] = self.house_center_y - (2.5 - all_shots['Vy']) * 8

            # Spatial coverage analysis
            spatial_metrics = {}

            for algo in ['Clustered', 'AllGrid']:
                algo_data = all_shots[all_shots['algorithm'] == algo]

                if not algo_data.empty:
                    x_coords = algo_data['estimated_x']
                    y_coords = algo_data['estimated_y']

                    # Coverage area (convex hull area approximation)
                    x_range = x_coords.max() - x_coords.min()
                    y_range = y_coords.max() - y_coords.min()
                    coverage_area = x_range * y_range

                    # Exploration density
                    total_area = (self.area_max_x * 2) * (self.area_max_y - 36.0)
                    density = len(algo_data) / total_area

                    # Centroid of exploration
                    centroid_x = x_coords.mean()
                    centroid_y = y_coords.mean()

                    # Spread metrics
                    spread_x = x_coords.std()
                    spread_y = y_coords.std()

                    # Distance from house center
                    distances_to_house = np.sqrt(
                        (x_coords - self.house_center_x)**2 +
                        (y_coords - self.house_center_y)**2
                    )

                    spatial_metrics[algo] = {
                        'coverage_area': coverage_area,
                        'exploration_density': density,
                        'centroid': (centroid_x, centroid_y),
                        'spread': (spread_x, spread_y),
                        'avg_distance_to_house': distances_to_house.mean(),
                        'min_distance_to_house': distances_to_house.min(),
                        'max_distance_to_house': distances_to_house.max()
                    }

            results['spatial_metrics'] = spatial_metrics

            # Zone occupation analysis
            zone_analysis = self.analyze_zone_preferences(all_shots)
            results['zone_analysis'] = zone_analysis

        return results

    def analyze_zone_preferences(self, shot_data: pd.DataFrame) -> Dict:
        """
        Analyze preferences for different strategic zones.

        Args:
            shot_data: DataFrame containing shot data with estimated positions

        Returns:
            Dictionary containing zone preference analysis
        """
        zone_preferences = {}

        for algo in ['Clustered', 'AllGrid']:
            algo_data = shot_data[shot_data['algorithm'] == algo]

            if not algo_data.empty:
                zone_counts = {}

                for _, shot in algo_data.iterrows():
                    x, y = shot['estimated_x'], shot['estimated_y']

                    # Check which zones this shot targets
                    for zone_name, zone_def in self.zones.items():
                        if zone_name in ['house', 'button', 'four_foot', 'eight_foot', 'twelve_foot']:
                            # Circular zones
                            center_x, center_y = zone_def['center']
                            radius = zone_def['radius']
                            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                            if distance <= radius:
                                zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

                        elif zone_name in ['guard_zone', 'back_line']:
                            # Rectangular zones
                            y_min, y_max = zone_def['y_range']
                            x_min, x_max = zone_def['x_range']

                            if y_min <= y <= y_max and x_min <= x <= x_max:
                                zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

                # Normalize to percentages
                total_shots = len(algo_data)
                zone_percentages = {zone: count/total_shots * 100
                                  for zone, count in zone_counts.items()}

                zone_preferences[algo] = zone_percentages

        return zone_preferences

    def identify_hotspots(self, shot_data: pd.DataFrame) -> Dict:
        """
        Identify spatial hotspots using clustering analysis.

        Args:
            shot_data: DataFrame containing shot data

        Returns:
            Dictionary containing hotspot analysis
        """
        hotspots = {}

        for algo in ['Clustered', 'AllGrid']:
            algo_data = shot_data[shot_data['algorithm'] == algo]

            if len(algo_data) >= 5:  # Minimum points for clustering
                # Extract coordinates
                coords = algo_data[['estimated_x', 'estimated_y']].values

                # Apply DBSCAN clustering to find hotspots
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)

                # Use DBSCAN to identify dense regions
                dbscan = DBSCAN(eps=0.3, min_samples=3)
                clusters = dbscan.fit_predict(coords_scaled)

                # Analyze clusters
                hotspot_info = []
                for cluster_id in set(clusters):
                    if cluster_id != -1:  # Ignore noise points
                        cluster_points = coords[clusters == cluster_id]
                        center_x = cluster_points[:, 0].mean()
                        center_y = cluster_points[:, 1].mean()
                        size = len(cluster_points)
                        spread = np.std(cluster_points, axis=0)

                        hotspot_info.append({
                            'cluster_id': cluster_id,
                            'center': (center_x, center_y),
                            'size': size,
                            'spread_x': spread[0],
                            'spread_y': spread[1]
                        })

                hotspots[algo] = {
                    'clusters': clusters,
                    'coordinates': coords,
                    'hotspot_info': hotspot_info,
                    'num_hotspots': len(hotspot_info)
                }

        return hotspots

    def analyze_exploration_efficiency(self, shot_data: pd.DataFrame) -> Dict:
        """
        Analyze exploration efficiency metrics.

        Args:
            shot_data: DataFrame containing shot data

        Returns:
            Dictionary containing efficiency analysis
        """
        efficiency = {}

        for algo in ['Clustered', 'AllGrid']:
            algo_data = shot_data[shot_data['algorithm'] == algo]

            if not algo_data.empty:
                coords = algo_data[['estimated_x', 'estimated_y']].values

                # Calculate pairwise distances
                if len(coords) > 1:
                    distances = pdist(coords)
                    distance_matrix = squareform(distances)

                    # Efficiency metrics
                    avg_distance = np.mean(distances)
                    min_distance = np.min(distances[distances > 0])
                    max_distance = np.max(distances)

                    # Space filling efficiency (how well distributed are the points)
                    # Using coefficient of variation of nearest neighbor distances
                    nn_distances = []
                    for i in range(len(coords)):
                        other_distances = distance_matrix[i][distance_matrix[i] > 0]
                        if len(other_distances) > 0:
                            nn_distances.append(np.min(other_distances))

                    if nn_distances:
                        space_filling_cv = np.std(nn_distances) / np.mean(nn_distances)
                    else:
                        space_filling_cv = 0

                    # Diversity index (based on spatial spread)
                    diversity_index = np.std(coords[:, 0]) * np.std(coords[:, 1])

                    efficiency[algo] = {
                        'avg_pairwise_distance': avg_distance,
                        'min_distance': min_distance,
                        'max_distance': max_distance,
                        'space_filling_cv': space_filling_cv,
                        'diversity_index': diversity_index,
                        'exploration_points': len(coords)
                    }

        return efficiency

    def create_spatial_visualizations(self, experiment_results: Dict, experiment_name: str) -> None:
        """
        Create comprehensive spatial exploration visualizations.

        Args:
            experiment_results: Results from analyze_experiment
            experiment_name: Name of the experiment for file naming
        """

        # 1. Spatial Density Heatmaps
        if 'shot_data' in experiment_results:
            shot_data = experiment_results['shot_data']

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Create ice surface outline
            def draw_ice_surface(ax):
                # House circles
                house = Circle((self.house_center_x, self.house_center_y), self.house_radius,
                             fill=False, color='blue', linewidth=2, alpha=0.7)
                button = Circle((self.house_center_x, self.house_center_y), 0.15,
                              fill=False, color='red', linewidth=2, alpha=0.7)
                four_foot = Circle((self.house_center_x, self.house_center_y), 0.61,
                                 fill=False, color='blue', linewidth=1, alpha=0.5)
                eight_foot = Circle((self.house_center_x, self.house_center_y), 1.22,
                                  fill=False, color='blue', linewidth=1, alpha=0.5)

                ax.add_patch(house)
                ax.add_patch(button)
                ax.add_patch(four_foot)
                ax.add_patch(eight_foot)

                # Ice boundaries
                ax.axhline(y=36.0, color='black', linewidth=2, alpha=0.8)
                ax.axhline(y=40.234, color='black', linewidth=2, alpha=0.8)
                ax.axvline(x=-2.375, color='black', linewidth=2, alpha=0.8)
                ax.axvline(x=2.375, color='black', linewidth=2, alpha=0.8)

                ax.set_xlim(-3, 3)
                ax.set_ylim(35, 41)
                ax.set_aspect('equal')

            # Clustered algorithm heatmap
            clustered_data = shot_data[shot_data['algorithm'] == 'Clustered']
            if not clustered_data.empty:
                x = clustered_data['estimated_x']
                y = clustered_data['estimated_y']

                axes[0,0].hexbin(x, y, gridsize=20, cmap='Blues', alpha=0.7)
                draw_ice_surface(axes[0,0])
                axes[0,0].set_title('Clustered MCTS - Spatial Density')
                axes[0,0].set_xlabel('X Position')
                axes[0,0].set_ylabel('Y Position')

            # AllGrid algorithm heatmap
            allgrid_data = shot_data[shot_data['algorithm'] == 'AllGrid']
            if not allgrid_data.empty:
                x = allgrid_data['estimated_x']
                y = allgrid_data['estimated_y']

                axes[0,1].hexbin(x, y, gridsize=20, cmap='Reds', alpha=0.7)
                draw_ice_surface(axes[0,1])
                axes[0,1].set_title('AllGrid MCTS - Spatial Density')
                axes[0,1].set_xlabel('X Position')
                axes[0,1].set_ylabel('Y Position')

            # Combined scatter plot with algorithm differentiation
            for algo, color in [('Clustered', 'blue'), ('AllGrid', 'red')]:
                algo_data = shot_data[shot_data['algorithm'] == algo]
                if not algo_data.empty:
                    axes[1,0].scatter(algo_data['estimated_x'], algo_data['estimated_y'],
                                    alpha=0.6, color=color, label=algo, s=30)

            draw_ice_surface(axes[1,0])
            axes[1,0].legend()
            axes[1,0].set_title('Combined Spatial Exploration')
            axes[1,0].set_xlabel('X Position')
            axes[1,0].set_ylabel('Y Position')

            # Zone preferences bar chart
            if 'zone_analysis' in experiment_results:
                zone_data = experiment_results['zone_analysis']
                zones = ['house', 'four_foot', 'eight_foot', 'guard_zone']

                clustered_prefs = [zone_data.get('Clustered', {}).get(zone, 0) for zone in zones]
                allgrid_prefs = [zone_data.get('AllGrid', {}).get(zone, 0) for zone in zones]

                x_pos = np.arange(len(zones))
                width = 0.35

                axes[1,1].bar(x_pos - width/2, clustered_prefs, width, label='Clustered', alpha=0.8)
                axes[1,1].bar(x_pos + width/2, allgrid_prefs, width, label='AllGrid', alpha=0.8)

                axes[1,1].set_xlabel('Strategic Zones')
                axes[1,1].set_ylabel('Percentage of Shots')
                axes[1,1].set_title('Zone Preference Comparison')
                axes[1,1].set_xticks(x_pos)
                axes[1,1].set_xticklabels(zones, rotation=45)
                axes[1,1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_spatial_exploration.png'))
            plt.close()

        # 2. Hotspot Analysis Visualization
        if 'hotspots' in experiment_results:
            hotspots = experiment_results['hotspots']

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            for i, (algo, color) in enumerate([('Clustered', 'blue'), ('AllGrid', 'red')]):
                if algo in hotspots:
                    hotspot_data = hotspots[algo]
                    coords = hotspot_data['coordinates']
                    clusters = hotspot_data['clusters']

                    # Plot points colored by cluster
                    unique_clusters = set(clusters)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

                    for cluster_id, cluster_color in zip(unique_clusters, colors):
                        if cluster_id == -1:
                            # Noise points
                            noise_mask = clusters == cluster_id
                            axes[i].scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                                          c='gray', alpha=0.5, s=20, label='Noise')
                        else:
                            cluster_mask = clusters == cluster_id
                            axes[i].scatter(coords[cluster_mask, 0], coords[cluster_mask, 1],
                                          c=[cluster_color], alpha=0.7, s=50,
                                          label=f'Hotspot {cluster_id}')

                    # Draw hotspot centers
                    for hotspot in hotspot_data['hotspot_info']:
                        center_x, center_y = hotspot['center']
                        axes[i].scatter(center_x, center_y, c='black', s=200, marker='X',
                                      edgecolors='white', linewidth=2)

                    draw_ice_surface(axes[i])
                    axes[i].set_title(f'{algo} MCTS - Hotspot Analysis')
                    axes[i].set_xlabel('X Position')
                    axes[i].set_ylabel('Y Position')
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_hotspots.png'))
            plt.close()

        # 3. Exploration Efficiency Comparison
        if 'exploration_efficiency' in experiment_results:
            efficiency = experiment_results['exploration_efficiency']

            metrics = ['avg_pairwise_distance', 'space_filling_cv', 'diversity_index']
            clustered_values = [efficiency.get('Clustered', {}).get(metric, 0) for metric in metrics]
            allgrid_values = [efficiency.get('AllGrid', {}).get(metric, 0) for metric in metrics]

            fig, ax = plt.subplots(figsize=(10, 6))

            x_pos = np.arange(len(metrics))
            width = 0.35

            ax.bar(x_pos - width/2, clustered_values, width, label='Clustered', alpha=0.8)
            ax.bar(x_pos + width/2, allgrid_values, width, label='AllGrid', alpha=0.8)

            ax.set_xlabel('Efficiency Metrics')
            ax.set_ylabel('Metric Value')
            ax.set_title('Spatial Exploration Efficiency Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['Avg Distance', 'Space Filling CV', 'Diversity Index'], rotation=45)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{experiment_name}_efficiency.png'))
            plt.close()

    def analyze_experiment(self, experiment_path: str, experiment_name: str) -> Dict:
        """
        Perform complete spatial exploration analysis on a single experiment.

        Args:
            experiment_path: Path to experiment directory
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing all analysis results
        """
        print(f"🗺️ Analyzing spatial exploration for: {experiment_name}")

        # Load spatial data
        spatial_data = self.load_spatial_data(experiment_path)

        if not any(not df.empty for df in spatial_data.values()):
            print("⚠️ No spatial data found for this experiment")
            return {}

        # Combine shot data
        all_shots = pd.concat([spatial_data['clustered_shots'], spatial_data['allgrid_shots']],
                            ignore_index=True)

        # Convert velocity to estimated positions
        all_shots['estimated_x'] = all_shots['Vx'] * 0.5
        all_shots['estimated_y'] = self.house_center_y - (2.5 - all_shots['Vy']) * 8

        results = {
            'experiment_name': experiment_name,
            'shot_data': all_shots,
            'data_summary': {
                'total_shots': len(all_shots),
                'clustered_shots': len(spatial_data['clustered_shots']),
                'allgrid_shots': len(spatial_data['allgrid_shots'])
            }
        }

        # Perform analyses
        results.update(self.calculate_spatial_metrics(spatial_data))
        results['hotspots'] = self.identify_hotspots(all_shots)
        results['exploration_efficiency'] = self.analyze_exploration_efficiency(all_shots)

        # Create visualizations
        self.create_spatial_visualizations(results, experiment_name)

        return results

    def run_spatial_analysis(self) -> Dict:
        """
        Run spatial exploration analysis on all discovered experiments.

        Returns:
            Dictionary containing results for all experiments
        """
        print("🚀 Starting Spatial Exploration Analysis")

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
                                print(f"✅ Completed spatial analysis for {dir_name}")
                        except Exception as e:
                            print(f"❌ Error analyzing {dir_name}: {e}")

        # Generate summary report
        self.generate_spatial_report(all_results)

        return all_results

    def generate_spatial_report(self, all_results: Dict) -> None:
        """
        Generate a comprehensive spatial exploration report.

        Args:
            all_results: Results from all experiments
        """
        report_path = os.path.join(self.output_dir, 'spatial_exploration_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SPATIAL EXPLORATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Experiments Analyzed: {len(all_results)}\n")
            f.write(f"Analysis Output Directory: {self.output_dir}\n\n")

            for exp_name, results in all_results.items():
                f.write(f"\n--- {exp_name} ---\n")

                if 'data_summary' in results:
                    summary = results['data_summary']
                    f.write(f"Total Shots Analyzed: {summary['total_shots']}\n")

                # Spatial metrics
                if 'spatial_metrics' in results:
                    spatial = results['spatial_metrics']
                    for algo, metrics in spatial.items():
                        f.write(f"{algo} Spatial Metrics:\n")
                        f.write(f"  Coverage Area: {metrics['coverage_area']:.3f}\n")
                        f.write(f"  Exploration Density: {metrics['exploration_density']:.6f}\n")
                        f.write(f"  Centroid: ({metrics['centroid'][0]:.3f}, {metrics['centroid'][1]:.3f})\n")
                        f.write(f"  Avg Distance to House: {metrics['avg_distance_to_house']:.3f}\n")

                # Hotspots
                if 'hotspots' in results:
                    hotspots = results['hotspots']
                    for algo, hotspot_data in hotspots.items():
                        f.write(f"{algo} Hotspots: {hotspot_data['num_hotspots']} identified\n")

                # Zone preferences
                if 'zone_analysis' in results:
                    zones = results['zone_analysis']
                    f.write("Zone Preferences:\n")
                    for algo, prefs in zones.items():
                        f.write(f"  {algo}: House={prefs.get('house', 0):.1f}%, ")
                        f.write(f"Guard Zone={prefs.get('guard_zone', 0):.1f}%\n")

        print(f"📊 Spatial exploration report generated: {report_path}")


def main():
    """Main function to run the spatial exploration analysis."""
    analyzer = SpatialExplorationAnalyzer()
    results = analyzer.run_spatial_analysis()

    print(f"\n🎉 Spatial exploration analysis complete!")
    print(f"🗺️ Analyzed {len(results)} experiments with detailed spatial patterns")
    print(f"📊 Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()