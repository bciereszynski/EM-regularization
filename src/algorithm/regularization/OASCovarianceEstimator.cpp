#include "OASCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> OASCovarianceEstimator::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    DoubleMatrix X = data;
    const int n = X.rows();
    const int d = X.cols();

    DoubleVector mu = get_mu(X, weights);
    translate_to_zero(X, mu);

    auto [covariance, _] = empirical(X, weights);
    double trace_mean = covariance.trace() / d;
    double alpha = covariance.array().square().mean();

    double num = alpha + trace_mean * trace_mean;
    double den = (n + 1.0) * (alpha - (trace_mean * trace_mean) / d);
    double shrinkage = (den == 0.0) ? 1.0 : std::min(num / den, 1.0);

    DoubleMatrix shrunk = shrunk_matrix(covariance, shrinkage);
    translate_to_mu(X, mu);

    return {shrunk.selfadjointView<Eigen::Lower>(), mu};
}

void OASCovarianceEstimator::translate_to_zero(DoubleMatrix& data, const DoubleVector& mu) {
    data.rowwise() -= mu.transpose();
}

void OASCovarianceEstimator::translate_to_mu(DoubleMatrix& data, const DoubleVector& mu) {
    data.rowwise() += mu.transpose();
}

std::pair<DoubleMatrix, DoubleVector> OASCovarianceEstimator::empirical(
    const DoubleMatrix& data, const std::vector<double>& weights) {
    const int n = data.rows();
    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), n);
    double sum_weights = w.sum();

    DoubleMatrix centered = data;
    DoubleMatrix weighted_centered = centered.array().colwise() * w.array();
    DoubleMatrix covariance = (weighted_centered.transpose() * centered) / sum_weights;

    return {covariance, DoubleVector::Zero(data.cols())};
}

DoubleMatrix OASCovarianceEstimator::shrunk_matrix(const DoubleMatrix& covariance, double shrinkage) {
    int d = covariance.cols();
    double trace_mean = covariance.trace() / d;
    DoubleMatrix shrunk = (1.0 - shrinkage) * covariance;
    for (int i = 0; i < d; ++i)
        shrunk(i, i) += shrinkage * trace_mean;
    return shrunk;
}
