#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <random>
#include <Eigen/Dense>

struct RowComparator {
    bool operator()(const std::vector<double> &a, const std::vector<double> &b) const {
        return a < b;
    }
};

std::pair<Eigen::MatrixXd, std::vector<int> > try_sampling_unique_data(
    std::mt19937 &rng,
    const Eigen::MatrixXd &data,
    int k
);

#endif //SAMPLING_H
