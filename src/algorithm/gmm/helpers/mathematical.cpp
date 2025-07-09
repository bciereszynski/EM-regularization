#include "mathematical.h"

#include <cmath>

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
    const std::vector<Eigen::MatrixXd> &precisionsCholesky) {
    const int n = data.rows();
    const int d = data.cols();
    Eigen::MatrixXd log_probabilities(n, k);
    const double log_2pi = std::log(2 * M_PI);
    const double d_log_2pi = d * log_2pi;

    Eigen::VectorXd log_det(k);
    for (int i = 0; i < k; ++i) {
        log_det(i) = precisionsCholesky[i].diagonal().array().log().sum();
    }

    for (int i = 0; i < k; ++i) {
        const Eigen::VectorXd mean = result.clusters.row(i).transpose();
        Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
        Eigen::MatrixXd transformed = centered * precisionsCholesky[i];
        log_probabilities.col(i) = transformed.rowwise().squaredNorm();
    }

    Eigen::VectorXd log_weights = Eigen::Map<const Eigen::VectorXd>(
        result.weights.data(), k).array().log();

    Eigen::MatrixXd result_matrix = (-0.5 * (d_log_2pi + log_probabilities.array())).matrix();
    result_matrix.rowwise() += log_det.transpose();
    result_matrix.rowwise() += log_weights.transpose();

    return result_matrix;
}
