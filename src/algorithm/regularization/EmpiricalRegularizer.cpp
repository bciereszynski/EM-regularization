#include "EmpiricalRegularizer.h"

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::compute_empirical(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.rows();

    Eigen::VectorXd mu = get_mu(data, weights);

    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), n);
    const double sum_weights = w.sum();

    DoubleMatrix centered = data.rowwise() - mu.transpose();
    const DoubleMatrix weighted_centered = centered.array().colwise() * w.array();
    DoubleMatrix covariance = (centered.transpose() * weighted_centered) / sum_weights;

    return {covariance, mu};
}

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    return compute_empirical(data, weights);
}
