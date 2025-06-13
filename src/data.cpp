#include "data.h"

#include <random>
#include <set>
#include <fstream>
#include <sstream>

std::vector<std::vector<double> > generate_data(const int n, const int d, const unsigned int seed) {
    std::vector<std::vector<double> > data(n, std::vector<double>(d));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return data;
}

std::vector<std::vector<double> > load_data_from_file(const std::string &path, std::vector<int> &expected_clusters,
                                                      int &k) {
    std::ifstream file(path);
    std::vector<std::vector<double> > data;
    std::set<int> unique_clusters;

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    expected_clusters.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;

        std::string token;
        bool first = true;
        while (std::getline(ss, token, ',')) {
            if (first) {
                first = false;
                auto cluster_id = std::stoi(token);
                unique_clusters.insert(cluster_id);
                expected_clusters.push_back(cluster_id - 1);
                continue;
            }
            row.push_back(std::stod(token));
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();

    k = unique_clusters.size();
    return data;
}
