#include "ShrunkCovarianceEstimator.h"


std::pair<DoubleMatrix, DoubleVector> ShrunkCovarianceEstimator::fit(
    const DoubleMatrix &data, const DoubleVector &weights) {
    return shrunk(data, weights, DEFAULT_SHRINKAGE);
}

std::pair<DoubleMatrix, DoubleVector> ShrunkCovarianceEstimator::shrunk(
    const DoubleMatrix &data, const DoubleVector &weights, const double shrinkage = DEFAULT_SHRINKAGE) {
    auto [covariance, mu] = compute_empirical(data, weights);

    return {shrunk_matrix(covariance, shrinkage), mu};
}

DoubleMatrix ShrunkCovarianceEstimator::shrunk_matrix(const DoubleMatrix &covariance, const double shrinkage) {
    const int d = covariance.cols();
    const double trace_mean = covariance.trace() / d;
    DoubleMatrix shrunk = (1.0 - shrinkage) * covariance;

    shrunk.diagonal().array() += shrinkage * trace_mean;
    return shrunk.selfadjointView<Eigen::Lower>();
}
