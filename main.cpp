#include <iostream>
#include <vector>

#include "data.h"
#include "algorithm/gmm.h"
#include "algorithm/gmmResult.h"

int main(int argc, char* argv[])  {
    constexpr int default_n = 100;
    constexpr int default_d = 2;
    constexpr int default_k = 2;
    constexpr unsigned int default_seed = 1;

    const int n = (argc > 1) ? std::stoi(argv[1]) : default_n;
    const int d = (argc > 2) ? std::stoi(argv[2]) : default_d;
    const int k = (argc > 3) ? std::stoi(argv[3]) : default_k;
    const unsigned int seed = (argc > 4) ? std::stoul(argv[4]) : default_seed;


    auto data = generate_data(n, d, seed);

    std::cout << "Data:" << std::endl;
    for (const auto& row : data) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    GMM gmm{};
    auto result = gmm.fit(data, k);

    std::cout << "Clusters:" << std::endl;
    for (const auto& row : result.clusters) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
