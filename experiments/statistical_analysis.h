#pragma once
#ifndef _STATISTICAL_ANALYSIS_H_
#define _STATISTICAL_ANALYSIS_H_

#include "efficiency_experiment.h"
#include <vector>
#include <string>
#include <map>

class StatisticalAnalysis {
public:
    struct StatisticalResult {
        double mean_efficiency_ratio;
        double std_efficiency_ratio;
        double median_efficiency_ratio;
        double p_value;
        int successful_experiments;
        int total_experiments;

        double clustered_success_rate;
        double allgrid_success_rate;

        std::vector<double> efficiency_ratios;
        double effect_size;

        double min_efficiency_ratio;
        double max_efficiency_ratio;
        double q1_efficiency_ratio;
        double q3_efficiency_ratio;
    };

    StatisticalResult analyzeResults(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results
    );

    void exportStatisticalSummary(
        const StatisticalResult& stats,
        const std::string& filename
    );

    void exportDetailedResults(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        const std::string& filename
    );

    void exportEfficiencyHistogram(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        const std::string& filename
    );

    void exportLearningCurves(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        const std::string& filename
    );

    void exportSuccessRateAnalysis(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        const std::string& filename
    );

private:
    double computeMean(const std::vector<double>& data);
    double computeStandardDeviation(const std::vector<double>& data);
    double computeMedian(std::vector<double> data);
    double computeQuartile(std::vector<double> data, double q);
    double performWilcoxonTest(const std::vector<double>& ratios);
    double performTTest(const std::vector<double>& ratios);
    double computeCohenD(const std::vector<double>& ratios);
    std::string getCurrentTimestamp();
};

#endif // _STATISTICAL_ANALYSIS_H_