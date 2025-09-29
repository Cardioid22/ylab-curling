#include "statistical_analysis.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <chrono>

StatisticalAnalysis::StatisticalResult StatisticalAnalysis::analyzeResults(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results) {

    std::cout << "\n=== Statistical Analysis ===" << std::endl;

    StatisticalResult stat_result;

    std::vector<double> efficiency_ratios;
    int successful_count = 0;
    int clustered_success_count = 0;
    int allgrid_success_count = 0;

    for (const auto& result : results) {
        if (result.clustered_success) clustered_success_count++;
        if (result.allgrid_success) allgrid_success_count++;

        if (result.clustered_success && result.allgrid_success && result.efficiency_ratio > 0) {
            efficiency_ratios.push_back(result.efficiency_ratio);
            successful_count++;
        }
    }

    stat_result.total_experiments = results.size();
    stat_result.successful_experiments = successful_count;
    stat_result.clustered_success_rate = static_cast<double>(clustered_success_count) / results.size();
    stat_result.allgrid_success_rate = static_cast<double>(allgrid_success_count) / results.size();
    stat_result.efficiency_ratios = efficiency_ratios;

    if (!efficiency_ratios.empty()) {
        stat_result.mean_efficiency_ratio = computeMean(efficiency_ratios);
        stat_result.std_efficiency_ratio = computeStandardDeviation(efficiency_ratios);
        stat_result.median_efficiency_ratio = computeMedian(efficiency_ratios);
        stat_result.p_value = performWilcoxonTest(efficiency_ratios);
        stat_result.effect_size = computeCohenD(efficiency_ratios);

        stat_result.min_efficiency_ratio = *std::min_element(efficiency_ratios.begin(), efficiency_ratios.end());
        stat_result.max_efficiency_ratio = *std::max_element(efficiency_ratios.begin(), efficiency_ratios.end());
        stat_result.q1_efficiency_ratio = computeQuartile(efficiency_ratios, 0.25);
        stat_result.q3_efficiency_ratio = computeQuartile(efficiency_ratios, 0.75);
    } else {
        stat_result.mean_efficiency_ratio = -1.0;
        stat_result.std_efficiency_ratio = -1.0;
        stat_result.median_efficiency_ratio = -1.0;
        stat_result.p_value = 1.0;
        stat_result.effect_size = 0.0;
        stat_result.min_efficiency_ratio = -1.0;
        stat_result.max_efficiency_ratio = -1.0;
        stat_result.q1_efficiency_ratio = -1.0;
        stat_result.q3_efficiency_ratio = -1.0;
    }

    std::cout << "Analysis completed:" << std::endl;
    std::cout << "  Successful experiments: " << successful_count << "/" << results.size() << std::endl;
    std::cout << "  Mean efficiency ratio: " << stat_result.mean_efficiency_ratio << std::endl;
    std::cout << "  P-value: " << stat_result.p_value << std::endl;

    return stat_result;
}

double StatisticalAnalysis::computeMean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double StatisticalAnalysis::computeStandardDeviation(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;

    double mean = computeMean(data);
    double sum_sq_diff = 0.0;

    for (double value : data) {
        double diff = value - mean;
        sum_sq_diff += diff * diff;
    }

    return std::sqrt(sum_sq_diff / (data.size() - 1));
}

double StatisticalAnalysis::computeMedian(std::vector<double> data) {
    if (data.empty()) return 0.0;

    std::sort(data.begin(), data.end());
    size_t size = data.size();

    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else {
        return data[size / 2];
    }
}

double StatisticalAnalysis::computeQuartile(std::vector<double> data, double q) {
    if (data.empty()) return 0.0;

    std::sort(data.begin(), data.end());
    double index = q * (data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return data[lower];
    } else {
        double weight = index - lower;
        return data[lower] * (1.0 - weight) + data[upper] * weight;
    }
}

double StatisticalAnalysis::performWilcoxonTest(const std::vector<double>& ratios) {
    // 簡易的な実装（1.0との比較）
    // 実際の実装では適切な統計ライブラリを使用することを推奨

    std::vector<double> differences;
    for (double ratio : ratios) {
        differences.push_back(ratio - 1.0);
    }

    // 符号ランク検定の簡易版
    int positive_count = 0;
    for (double diff : differences) {
        if (diff > 0) positive_count++;
    }

    double proportion = static_cast<double>(positive_count) / differences.size();

    // 簡易的なp値計算（正確には二項分布を使用）
    if (proportion < 0.5) {
        return 2.0 * proportion;
    } else {
        return 2.0 * (1.0 - proportion);
    }
}

double StatisticalAnalysis::performTTest(const std::vector<double>& ratios) {
    // t検定の簡易実装
    if (ratios.size() < 2) return 1.0;

    double mean = computeMean(ratios);
    double std_dev = computeStandardDeviation(ratios);
    double se = std_dev / std::sqrt(ratios.size());

    double t_stat = (mean - 1.0) / se;

    // 簡易的なp値（実際にはt分布を使用）
    return 2.0 * (1.0 - std::abs(t_stat) / (std::abs(t_stat) + ratios.size() - 1));
}

double StatisticalAnalysis::computeCohenD(const std::vector<double>& ratios) {
    double mean = computeMean(ratios);
    double std_dev = computeStandardDeviation(ratios);

    if (std_dev == 0.0) return 0.0;

    return (1.0 - mean) / std_dev; // 1.0を基準とした効果サイズ
}

std::string StatisticalAnalysis::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

void StatisticalAnalysis::exportStatisticalSummary(
    const StatisticalResult& stats,
    const std::string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Metric,Value\n";
    file << "Mean Efficiency Ratio," << stats.mean_efficiency_ratio << "\n";
    file << "Std Efficiency Ratio," << stats.std_efficiency_ratio << "\n";
    file << "Median Efficiency Ratio," << stats.median_efficiency_ratio << "\n";
    file << "Min Efficiency Ratio," << stats.min_efficiency_ratio << "\n";
    file << "Max Efficiency Ratio," << stats.max_efficiency_ratio << "\n";
    file << "Q1 Efficiency Ratio," << stats.q1_efficiency_ratio << "\n";
    file << "Q3 Efficiency Ratio," << stats.q3_efficiency_ratio << "\n";
    file << "P-value," << stats.p_value << "\n";
    file << "Effect Size (Cohen's d)," << stats.effect_size << "\n";
    file << "Successful Experiments," << stats.successful_experiments << "\n";
    file << "Total Experiments," << stats.total_experiments << "\n";
    file << "Success Rate," << static_cast<double>(stats.successful_experiments) / stats.total_experiments << "\n";
    file << "Clustered Success Rate," << stats.clustered_success_rate << "\n";
    file << "AllGrid Success Rate," << stats.allgrid_success_rate << "\n";

    file.close();
    std::cout << "Statistical summary saved to: " << filename << std::endl;
}

void StatisticalAnalysis::exportDetailedResults(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    const std::string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Experiment,Clustered_Success,AllGrid_Success,Clustered_Iterations,AllGrid_Iterations,Efficiency_Ratio,Clustered_Score,AllGrid_Score\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        file << i << ","
             << (result.clustered_success ? 1 : 0) << ","
             << (result.allgrid_success ? 1 : 0) << ","
             << result.clustered_discovery_iterations << ","
             << result.allgrid_discovery_iterations << ","
             << result.efficiency_ratio << ","
             << result.clustered_final_score << ","
             << result.allgrid_final_score << "\n";
    }

    file.close();
    std::cout << "Detailed results saved to: " << filename << std::endl;
}

void StatisticalAnalysis::exportEfficiencyHistogram(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    const std::string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Efficiency_Ratio\n";

    for (const auto& result : results) {
        if (result.clustered_success && result.allgrid_success && result.efficiency_ratio > 0) {
            file << result.efficiency_ratio << "\n";
        }
    }

    file.close();
    std::cout << "Efficiency histogram data saved to: " << filename << std::endl;
}

void StatisticalAnalysis::exportLearningCurves(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    const std::string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Experiment,Iteration,Clustered_Score,AllGrid_Score\n";

    for (size_t exp = 0; exp < results.size(); ++exp) {
        const auto& result = results[exp];

        if (!result.clustered_success || !result.allgrid_success) continue;

        size_t max_iterations = std::min(result.clustered_score_history.size(),
                                       result.allgrid_score_history.size());

        for (size_t iter = 0; iter < max_iterations; ++iter) {
            file << exp << "," << iter << ","
                 << result.clustered_score_history[iter] << ","
                 << result.allgrid_score_history[iter] << "\n";
        }
    }

    file.close();
    std::cout << "Learning curves saved to: " << filename << std::endl;
}

void StatisticalAnalysis::exportSuccessRateAnalysis(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    const std::string& filename) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "Method,Success_Count,Total_Count,Success_Rate\n";

    int clustered_success = 0, allgrid_success = 0;
    for (const auto& result : results) {
        if (result.clustered_success) clustered_success++;
        if (result.allgrid_success) allgrid_success++;
    }

    file << "Clustered," << clustered_success << "," << results.size() << ","
         << static_cast<double>(clustered_success) / results.size() << "\n";
    file << "AllGrid," << allgrid_success << "," << results.size() << ","
         << static_cast<double>(allgrid_success) / results.size() << "\n";

    file.close();
    std::cout << "Success rate analysis saved to: " << filename << std::endl;
}