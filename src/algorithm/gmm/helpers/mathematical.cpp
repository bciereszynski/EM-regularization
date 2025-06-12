#include "mathematical.h"

#include <cmath>
#include <limits>

Eigen::VectorXd log_sum_exp(const Eigen::MatrixXd &m) {
    Eigen::VectorXd max_vals = m.rowwise().maxCoeff();

    Eigen::VectorXd results(m.rows());
    for (int i = 0; i < m.rows(); ++i) {
        if (std::isinf(max_vals[i])) {
            results[i] = max_vals[i];
        } else {
            results[i] = max_vals[i] + std::log((m.row(i).array() - max_vals[i]).exp().sum());
        }
    }
    return results;
}


Eigen::MatrixXd estimate_weighted_log_probabilities(
    const Eigen::MatrixXd &data, const int k, const GMMResult &result,
    const std::vector<Eigen::MatrixXd> &precisionsCholesky
) {
    const int n = data.rows();
    const int d = data.cols();

    Eigen::MatrixXd log_probabilities(n, k);
    const double constant_term = -0.5 * d * std::log(2.0 * M_PI);

    for (int i = 0; i < k; ++i) {
        const Eigen::VectorXd mean = result.clusters.row(i).transpose();
        const Eigen::MatrixXd &L = precisionsCholesky[i];
        const double log_det_precision = L.diagonal().array().log().sum();

        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        Eigen::MatrixXd transformed = centered * L;
        Eigen::VectorXd sq_norm = transformed.rowwise().squaredNorm();

        log_probabilities.col(i) = constant_term + log_det_precision
                                   - 0.5 * sq_norm.array()
                                   + std::log(result.weights[i]);
    }

    return log_probabilities;
}
