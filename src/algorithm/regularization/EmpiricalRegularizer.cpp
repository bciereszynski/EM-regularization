#include "EmpiricalRegularizer.h"

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::compute_empirical(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.rows();

    Eigen::Map<const Eigen::VectorXd> w(weights.data(), n);
    const double sum_weights = w.sum();

    DoubleVector mu = (data.transpose() * w) / sum_weights;
    DoubleMatrix centered = data.rowwise() - mu.transpose();
    const DoubleMatrix weighted_centered = centered.array().colwise() * w.array();
    DoubleMatrix covariance = (centered.transpose() * weighted_centered) / sum_weights;

    covariance = 0.5 * (covariance + covariance.transpose());
    covariance.diagonal().array() += 1e-6;

    return {covariance, mu};
}

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    return compute_empirical(data, weights);
}
