#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <random>

struct RowComparator {
    bool operator()(const std::vector<double> &a, const std::vector<double> &b) const {
        return a < b;
    }
};

std::pair<std::vector<std::vector<double> >, std::vector<int> > try_sampling_unique_data(
    std::mt19937 &rng,
    const std::vector<std::vector<double> > &data,
    int k
);

#endif //SAMPLING_H
