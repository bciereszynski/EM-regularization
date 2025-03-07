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

std::vector<std::vector<double> > load_data_from_file(const std::string &path, int &k) {
    std::ifstream file(path);
    std::vector<std::vector<double> > data;
    std::set<int> clusters;

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;

        std::string token;
        bool first = true;
        while (std::getline(ss, token, ',')) {
            if (first) {
                first = false;
                clusters.insert(std::stoi(token));
                continue;
            }
            row.push_back(std::stod(token));
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();

    k = clusters.size();
    return data;
}
