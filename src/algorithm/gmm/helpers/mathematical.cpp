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

    Eigen::VectorXd log_det(k);
    for (int i = 0; i < k; ++i) {
        log_det(i) = precisionsCholesky[i].diagonal().array().log().sum();
    }
    for (int i = 0; i < k; ++i) {

        Eigen::MatrixXd transformed = data * precisionsCholesky[i];
        const Eigen::RowVectorXd mean_transformed = result.clusters.row(i) * precisionsCholesky[i];
        transformed.rowwise() -= mean_transformed;

        Eigen::VectorXd sq_norms = transformed.rowwise().squaredNorm();

        log_probabilities.col(i) =
                (-0.5 * d * log_2pi + log_det(i) + std::log(result.weights[i]))
                - 0.5 * sq_norms.array();
    }

    return log_probabilities;
}
