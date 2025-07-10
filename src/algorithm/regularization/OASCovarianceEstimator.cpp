#include "OASCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> OASCovarianceEstimator::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.rows();
    const int d = data.cols();

    DoubleVector mu = get_mu(data, weights);

    DoubleMatrix X = data.rowwise() - mu.transpose();

    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), weights.size());
    const double sum_weights = w.sum();
    DoubleMatrix weighted_centered = X.array().colwise() * w.array();
    DoubleMatrix covariance = (weighted_centered.transpose() * X) / sum_weights;

    const double trace_mean = covariance.trace() / d;
    const double alpha = covariance.array().square().mean();

    const double num = alpha + trace_mean * trace_mean;
    const double den = (n + 1.0) * (alpha - (trace_mean * trace_mean) / d);
    const double shrinkage = (den == 0.0) ? 1.0 : std::min(num / den, 1.0);

    DoubleMatrix shrunk = shrunk_matrix(covariance, shrinkage);

    return {shrunk, mu};
}
