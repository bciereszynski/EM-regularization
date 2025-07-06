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

int main(const int argc, char *argv[]) {
    std::string directory = "test_data";
    int k;
    int seed = 123;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            directory = argv[++i];
        }
        if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        }
    }


    std::vector<int> expected_clusters;

    GeneticalAlgorithm ga{std::mt19937(seed), 250, 150, 40, 50, false};
    std::cout << "precompile ";
    std::vector<std::vector<double> > data = load_data_from_file(R"(data/3_2_-0.26_1.csv)", expected_clusters, k);

    GMMResult result = ga.run(data, k);
    print_result(result);

    std::ofstream output_file("evaluation/results.txt");

    for (const auto &entry: fs::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") {
            continue;
        }
        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();
        data = load_data_from_file(path, expected_clusters, k);

        auto *empirical = new EmpiricalRegularizer();
        try {
            result = ga.run(data, k, empirical);
            print_result(result);
            output_file << "gmm_hg " << filename << " "
                    << result.objective << " "
                    << result.iterations << " "
                    << result.elapsed << " ";
            for (int val: result.assignments) output_file << val << ",";
            output_file << " ";
            for (int val: expected_clusters) output_file << val << ",";
            output_file << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        auto *shrunk = new ShrunkCovarianceEstimator();
        try {
            result = ga.run(data, k, shrunk);
            print_result(result);
            output_file << "gmm_hg_shrunk " << filename << " "
                    << result.objective << " "
                    << result.iterations << " "
                    << result.elapsed << " ";
            for (int val: result.assignments) output_file << val << ",";
            output_file << " ";
            for (int val: expected_clusters) output_file << val << ",";
            output_file << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }


        auto *LW = new LedoitWolfCovarianceEstimator();
        try {
            result = ga.run(data, k, LW);
            print_result(result);
            output_file << "gmm_hg_ledoitwolf " << filename << " "
                    << result.objective << " "
                    << result.iterations << " "
                    << result.elapsed << " ";
            for (int val: result.assignments) output_file << val << ",";
            output_file << " ";
            for (int val: expected_clusters) output_file << val << ",";
            output_file << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        auto *OAS = new OASCovarianceEstimator();
        try {
            result = ga.run(data, k, OAS);
            print_result(result);
            output_file << "gmm_hg_oas " << filename << " "
                    << result.objective << " "
                    << result.iterations << " "
                    << result.elapsed << " ";
            for (int val: result.assignments) output_file << val << ",";
            output_file << " ";
            for (int val: expected_clusters) output_file << val << ",";
            output_file << std::endl;
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    output_file.close();
    return 0;
}
