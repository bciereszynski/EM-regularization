#ifndef MATHEMATICAL_H
#define MATHEMATICAL_H

#include <vector>
#include "../gmmResult.h"

template<typename T>
T one(const T &x) {
    return static_cast<T>(1);
}

template<typename T>
T zero(const T &x) {
    return static_cast<T>(0);
}

double log_sum_exp(const std::vector<double> &probabilities);

std::vector<std::vector<double> > estimate_weighted_log_probabilities(
    const Eigen::MatrixXd &data, int k, const GMMResult &result,
    const std::vector<Eigen::MatrixXd> &precisionsCholesky
);

#endif //MATHEMATICAL_H
