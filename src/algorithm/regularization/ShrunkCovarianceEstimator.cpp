#include "ShrunkCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> ShrunkCovarianceEstimator::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    auto [covariance, mu] = EmpiricalRegularizer::fit(data, weights);

    DoubleMatrix shrunk = shrunk_matrix(covariance, DEFAULT_SHRINKAGE);
    return {shrunk.selfadjointView<Eigen::Lower>(), mu};
}

DoubleMatrix ShrunkCovarianceEstimator::shrunk_matrix(const DoubleMatrix& covariance, double shrinkage) {
    int d = covariance.cols();
    double trace_mean = covariance.trace() / d;

    DoubleMatrix shrunk = (1.0 - shrinkage) * covariance;
    for (int i = 0; i < d; ++i) {
        shrunk(i, i) += shrinkage * trace_mean;
    }
    return shrunk;
}