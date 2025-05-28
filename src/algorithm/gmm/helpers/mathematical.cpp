#include "mathematical.h"

#include <cmath>
#include <limits>

double log_sum_exp(const Eigen::Ref<const Eigen::VectorXd> &values) {
    const double max_val = values.maxCoeff();

    if (std::isinf(max_val)) return max_val;

    return max_val + std::log((values.array() - max_val).exp().sum());
}


Eigen::MatrixXd estimate_weighted_log_probabilities(
    const Eigen::MatrixXd &data, const int k, const GMMResult &result,
    const std::vector<Eigen::MatrixXd> &precisionsCholesky
) {
    const int n = data.rows();
    const int d = data.cols();

    Eigen::MatrixXd log_probabilities(n, k);

    for (int i = 0; i < k; ++i) {
        const Eigen::VectorXd mean = result.clusters.row(i).transpose();
        const Eigen::MatrixXd &cholesky = precisionsCholesky[i];

        const double log_det = cholesky.diagonal().array().log().sum();

        Eigen::MatrixXd y = (data.rowwise() - mean.transpose()) * cholesky;

        Eigen::VectorXd prob = y.rowwise().squaredNorm();

        log_probabilities.col(i) = (-0.5 * (d * std::log(2 * M_PI) + prob.array()) +
                                    log_det + std::log(result.weights[i])).matrix();
    }

    return log_probabilities;
}
