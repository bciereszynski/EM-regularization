#include "EmpiricalRegularizer.h"

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::compute_empirical(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.rows();
    const int d = data.cols();

    Eigen::Map<const Eigen::VectorXd> w(weights.data(), n);
    const double sum_weights = w.sum();

    if (sum_weights < std::numeric_limits<double>::min()) {
        return {DoubleMatrix::Identity(d, d) * 1e-6, DoubleVector::Zero(d)};
    }

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
