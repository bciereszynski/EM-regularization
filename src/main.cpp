#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>

#include "data.h"
#include "algorithm/GeneticalAlgorithm.h"
#include "algorithm/gmm/gmm.h"
#include "algorithm/gmm/gmmResult.h"

#include "algorithm/regularization/ShrunkCovarianceEstimator.h"
#include "algorithm/regularization/EmpiricalRegularizer.h"
#include "algorithm/regularization/LedoitWolfCovarianceEstimator.h"
#include "algorithm/regularization/OASCovarianceEstimator.h"

namespace fs = std::filesystem;

void print_result(GMMResult result) {
    // std::cout << "Clusters:" << std::endl;
    // std::cout << result.clusters << std::endl;
    std::cout << result.objective << " " << result.iterations << " " << result.elapsed << std::endl;

    // for (const int assignment: result.assignments) {
    //     std::cout << assignment << std::endl;
    // }
}

void run_and_log(GeneticalAlgorithm &ga,
                 const std::vector<std::vector<double> > &data,
                 const int k,
                 const std::string &filename,
                 std::ofstream &output_file,
                 const std::vector<int> &expected_clusters,
                 CovarianceMatrixRegularizer *estimator,
                 const std::string &flag) {
    GMMResult result = ga.run(data, k, estimator);
    print_result(result);

    output_file << flag << " " << filename << " "
            << result.objective << " "
            << result.iterations << " "
            << result.elapsed << " ";

    output_file << "[";
    for (const int val: result.assignments) output_file << val << ",";
    output_file << "] ";

    output_file << "[";
    for (const int val: expected_clusters) output_file << val << ",";
    output_file << "] ";
    output_file << std::endl;
}

void run_multiple(const std::vector<GeneticalAlgorithm *> &gas,
                  const std::vector<std::vector<double> > &data, const int k,
                  const std::string &filename, std::ofstream &output_file,
                  const std::vector<int> &expected_clusters, CovarianceMatrixRegularizer *estimator,
                  const std::string &flag) {
    for (GeneticalAlgorithm *ga: gas) {
        run_and_log(*ga, data, k, filename, output_file, expected_clusters, estimator, flag);
    }
}

int main(const int argc, char *argv[]) {
    Eigen::setNbThreads(1);

    std::string directory = "test_data";
    int k;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            directory = argv[++i];
        }
    }
    int max_iterations = 250;
    int max_iterations_without_improvement = 150;
    int pop_min_size = 40;
    int pop_max_size = 50;


    std::vector<int> expected_clusters;

    std::vector<GeneticalAlgorithm> gas = {
        GeneticalAlgorithm{
            std::mt19937(123), max_iterations, max_iterations_without_improvement, pop_min_size, pop_max_size, false
        },
        GeneticalAlgorithm{
            std::mt19937(42), max_iterations, max_iterations_without_improvement, pop_min_size, pop_max_size, false
        },
        GeneticalAlgorithm{
            std::mt19937(33), max_iterations, max_iterations_without_improvement, pop_min_size, pop_max_size, false
        }
    };

    std::vector<std::vector<double> > data;

    auto empirical = EmpiricalRegularizer();
    auto shrunk = ShrunkCovarianceEstimator();
    auto LW = LedoitWolfCovarianceEstimator();
    auto OAS = OASCovarianceEstimator();
    std::cout << "precompile\n";
    data = load_data_from_file(R"(data/3_2_-0.26_1.csv)", expected_clusters, k);
    for (GeneticalAlgorithm &ga: gas) {
        ga.run(data, k, &empirical);
        ga.run(data, k, &shrunk);
        ga.run(data, k, &LW);
        ga.run(data, k, &OAS);
    }

    std::vector<GeneticalAlgorithm *> gas_ptrs;
    for (auto &ga: gas) {
        gas_ptrs.push_back(&ga);
    }

    std::ofstream output_file("evaluation/results.txt");

    int total_files = 0;
    for (const auto &entry: fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            total_files++;
        }
    }
    int processed_files = 0;

    for (const auto &entry: fs::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") {
            continue;
        }

        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        data = load_data_from_file(path, expected_clusters, k);

        processed_files++;
        std::cout << "Processing " << filename
                << " (" << processed_files << "/" << total_files << ")\n";

        try {
            run_multiple(gas_ptrs, data, k, filename, output_file, expected_clusters, &empirical,
                         "gmm_hg");
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        try {
            run_multiple(gas_ptrs, data, k, filename, output_file, expected_clusters, &shrunk,
                         "gmm_hg_shrunk");
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        try {
            run_multiple(gas_ptrs, data, k, filename, output_file, expected_clusters, &LW,
                         "gmm_hg_ledoitwolf");
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        try {
            run_multiple(gas_ptrs, data, k, filename, output_file, expected_clusters, &OAS,
                         "gmm_hg_oas");
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    output_file.close();
    return 0;
}
