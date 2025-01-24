#include "data.h"
#include <random>

std::vector<std::vector<double>> generate_data(const int n, const int d, const unsigned int seed) {
    std::vector<std::vector<double>> data(n, std::vector<double>(d));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return data;
}