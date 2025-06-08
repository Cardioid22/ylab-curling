#pragma once
#ifndef _CLUSTERING_H
#define _CLUSTERING_H
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include <iostream>
#include <set>

namespace dc = digitalcurling3;

class Clustering {
public:
Clustering();
int n_desired_clusters = 4;
std::vector<dc::GameState> states;
std::vector<std::set<int>> clusters;
LinkageMatrix linkage;
float dist(dc::GameState const& a, dc::GameState const& b);
std::vector<std::vector<float>> MakeDistanceTable(std::vector<dc::GameState> const& states);
std::tuple<int, int, float> findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters);
LinkageMatrix hierarchicalClustering(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters, int n_desired_clusters = 4);
std::vector<std::set<int>> getClusters();
std::vector<int> getRecommndedStates(std::vector<std::set<int>> clusters);
private:
	const float HouseRadius = 1.829;
	const float AreaMaxX = 2.375;
	const float AreaMaxY = 40.234;
	const float HouseCenterX = 0;
	const float HouseCenterY = 38.405;
	int GridSize_M = 4;
	int GridSize_N = 4;
};

#endif