#include "EmpiricalRegularizer.h"

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::compute_empirical(
    const DoubleMatrix &data, const DoubleVector &weights) {
    const double sum_weights = weights.sum();

    DoubleMatrix centered(data.rows(), data.cols());
    DoubleMatrix weighted_centered(data.rows(), data.cols());

    DoubleVector mu = get_mu(data, weights);
    centered = data.rowwise() - mu.transpose();
    weighted_centered.array() = centered.array().colwise() * weights.array();
    DoubleMatrix covariance = (centered.transpose() * weighted_centered) / sum_weights;

    covariance = 0.5 * (covariance + covariance.transpose());
    covariance.diagonal().array() += EPS;

    return {covariance, mu};
}

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const DoubleVector &weights) {
    return compute_empirical(data, weights);
}
