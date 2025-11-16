#pragma once
#ifndef _CLUSTERING_H
#define _CLUSTERING_H
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "analysis.h"
#include <iostream>
#include <set>
#include <vector>
#include <tuple>

namespace dc = digitalcurling3;

class Clustering {
public:
	Clustering(int k_clusters, std::vector<dc::GameState> all_states, int gridM, int gridN, dc::Team team);

	std::vector<std::set<int>> getClusters();
	std::vector<int> getRecommendedStates();
	std::vector<std::vector<int>> get_clusters_id_table();

private:
	const float HouseRadius_ = 1.829;
	const float AreaMaxX_ = 2.375;
	const float AreaMaxY_ = 40.234;
	const float HouseCenterX_ = 0;
	const float HouseCenterY_ = 38.405;
	int GridSize_M_ = 10;
	int GridSize_N_ = 10;
	int n_desired_clusters = 6;
	bool cluster_exists = false;
	dc::Team g_team;

	std::vector<dc::GameState> states;
	std::vector<std::set<int>> clusters;
	std::vector<std::vector<int>> recommend_states;
	LinkageMatrix linkage;

	bool IsInHouse(const std::optional<dc::Transform>& stone) const;
	std::vector<std::pair<size_t, size_t>> SortStones(const std::array<std::array<std::optional<dc::Transform>, 8>, 2>& all_stones) const;
	float dist(dc::GameState const& a, dc::GameState const& b) const;
	std::vector<std::vector<float>> MakeDistanceTable(std::vector<dc::GameState> const& states);
	std::tuple<int, int, float> findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters);
    LinkageMatrix hierarchicalClustering(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters, int n_desired_clusters);
	std::vector<std::vector<int>> calculateMedioid(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters);
	float EvaluateScoreBoard(dc::GameState const& state);
};

#endif