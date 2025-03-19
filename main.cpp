#include <iostream>
#include <vector>

#include "data.h"
#include "algorithm/Evaluator.h"
#include "algorithm/gmm/gmm.h"
#include "algorithm/gmm/gmmResult.h"

int main(const int argc, char *argv[]) {
    std::string path = R"(test_data/3_2_0.01_1.csv)";
    int k = 2;
    int d = 2;
    int n = 100;
    int seed = 0;

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
        data = generate_data(n, d, seed);
    } else {
        data = load_data_from_file(path, expected_clusters, k);
    }

    GMM gmm{};
    auto result = gmm.fit(data, k);

    std::cout << "Clusters:" << std::endl;
    for (const auto &row: result.clusters) {
        for (const auto &value: row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Evaluation:" << std::endl;

    Evaluator evaluator(data, expected_clusters, k, result);
    std::cout << "centroid_index_sphere: " << evaluator.centroid_index_sphere() << std::endl;

    return 0;
}
