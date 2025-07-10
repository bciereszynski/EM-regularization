#include "OASCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> OASCovarianceEstimator::fit(
    const DoubleMatrix &data, const DoubleVector &weights) {
    const int n = data.rows();
    const int d = data.cols();

    DoubleVector mu = get_mu(data, weights);
    DoubleMatrix X = data.rowwise() - mu.transpose();

    DoubleMatrix covariance;
    std::tie(covariance, std::ignore) = compute_empirical(data, weights);

    const double trace_mean = covariance.diagonal().sum() / d;
    const double alpha = covariance.squaredNorm() / (d * d);

    const double trace_sq = trace_mean * trace_mean;
    const double num = alpha + trace_sq;
    const double den = (n + 1.0) * (alpha - (trace_sq) / d);
    const double shrinkage = (abs(den) <= EPS) ? 1.0 : std::clamp(num / den, 0.0, 1.0);

    DoubleMatrix shrunk = shrunk_matrix(covariance, shrinkage);

    return {shrunk, mu};
}
