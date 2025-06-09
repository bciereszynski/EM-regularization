#include <iostream>
#include <vector>
#include <filesystem>

#include "data.h"
#include "algorithm/GeneticalAlgorithm.h"
#include "algorithm/gmm/gmm.h"
#include "algorithm/gmm/gmmResult.h"


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

    std::vector<std::vector<double> > data;
    std::vector<int> expected_clusters;

    GeneticalAlgorithm ga{std::mt19937(seed), 200, 150, 40, 50, false};
    std::cout << "precompile ";
    data = load_data_from_file(R"(data/3_2_-0.26_1.csv)", expected_clusters, k);
    const auto result = ga.run(data, k);
    print_result(result);
    for (const auto &entry: fs::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") {
            continue;
        }
        std::string path = entry.path().string();
        data = load_data_from_file(path, expected_clusters, k);

        GMMResult result_i(d, data.size(), k);
        std::cout << path << " ";
        GMMResult result_i(data[0].size(), data.size(), k);
        try {
            result_i = ga.run(data, k);
        } catch (std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        print_result(result_i);
    }

    return 0;
}
