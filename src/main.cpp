#include <iostream>
#include <vector>

#include "data.h"
#include "algorithm/GeneticalAlgorithm.h"
#include "algorithm/gmm/gmm.h"
#include "algorithm/gmm/gmmResult.h"

int main(const int argc, char *argv[]) {
    std::string path = R"(test_data/3_2_-0.26_1.csv)";
    int k = 2;
    int d = 2;
    int n = 100;
    int seed = 123;

    for (int i = 1; i < argc; i++) {
        if (std::string arg = argv[i]; arg == "--k" && i + 1 < argc) {
            k = std::stoi(argv[++i]);
        } else if (arg == "--path" && i + 1 < argc) {
            path = argv[++i];
            std::cerr << "Path entered - other options will be ignored" << std::endl;
            break;
        } else if (arg == "--n" && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if (arg == "--d" && i + 1 < argc) {
            d = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        }
    }

    std::vector<std::vector<double> > data;
    std::vector<int> expected_clusters;

    if (path.empty()) {
        return 1;
    } else {
        data = load_data_from_file(path, expected_clusters, k);
    }

    GeneticalAlgorithm ga{std::mt19937(seed), 500, 150, 40, 50, false};
    const auto result = ga.run(data, k);

    std::cout << "Clusters:" << std::endl;
    std::cout << result.clusters << std::endl;
    std::cout << result.iterations << std::endl;
    std::cout << result.elapsed << std::endl;
    std::cout << result.objective << std::endl;

    // for (const int assignment: result.assignments) {
    //     std::cout << assignment << std::endl;
    // }

    return 0;
}
