#include "EmpiricalRegularizer.h"

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::compute_empirical(
    const DoubleMatrix &data, const DoubleVector &weights) {
    const double sum_weights = weights.sum();

    DoubleVector mu = get_mu(data, weights);
    DoubleMatrix centered = data.rowwise() - mu.transpose();
    const DoubleMatrix weighted_centered = centered.array().colwise() * weights.array();
    DoubleMatrix covariance = (centered.transpose() * weighted_centered) / sum_weights;

    covariance = 0.5 * (covariance + covariance.transpose());
    covariance.diagonal().array() += EPS;

    return {covariance, mu};
}

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const DoubleVector &weights) {
    return compute_empirical(data, weights);
}
