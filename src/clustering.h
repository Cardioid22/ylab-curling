#pragma once
#ifndef _CLUSTERING_H
#define _CLUSTERING_H
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include <iostream>
#include <set>
#include <vector>
#include <tuple>

namespace dc = digitalcurling3;

class Clustering {
public:
	Clustering(int k_clusters, std::vector<dc::GameState> all_states);

	std::vector<std::set<int>> getClusters(void);
	std::vector<int> getRecommendedStates(std::vector<std::set<int>> clusters);

private:
	const float HouseRadius_ = 1.829;
	const float AreaMaxX_ = 2.375;
	const float AreaMaxY_ = 40.234;
	const float HouseCenterX_ = 0;
	const float HouseCenterY_ = 38.405;
	const int GridSize_M_ = 4;
	const int GridSize_N_ = 4;
	int n_desired_clusters = 4;

	std::vector<dc::GameState> states;
	std::vector<std::set<int>> clusters;
	LinkageMatrix linkage;

	float dist(dc::GameState const& a, dc::GameState const& b);
	std::vector<std::vector<float>> MakeDistanceTable(std::vector<dc::GameState> const& states);
	std::tuple<int, int, float> findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters);
    LinkageMatrix hierarchicalClustering(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters, int n_desired_clusters = 4);
};

#endif