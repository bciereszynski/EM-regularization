#include "LedoitWolfCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> LedoitWolfCovarianceEstimator::fit(
const DoubleMatrix &data, const std::vector<double> &weights) {
    DoubleMatrix X = data;
    const int n = X.rows();
    const int d = X.cols();

    DoubleVector mu = get_mu(X, weights);
    translate_to_zero(X, mu);

    DoubleMatrix X2 = X.array().square();
    DoubleVector trace = X2.colwise().sum() / n;
    double trace_mean = trace.sum() / d;

    double beta_coeff = (X2.transpose() * X2).sum();
    double delta_coeff = (X.transpose() * X).array().square().sum() / (n * n);

    double beta = 1.0 / (n * d) * (beta_coeff / n - delta_coeff);
    double delta = delta_coeff - 2.0 * trace_mean * trace.sum() + d * trace_mean * trace_mean;
    delta /= d;

    beta = std::min(beta, delta);
    double shrinkage = (beta == 0.0) ? 0.0 : beta / delta;

    translate_to_mu(X, mu);

    auto [cov, _] = EmpiricalRegularizer::fit(X, weights);
    DoubleMatrix shrunk = shrunk_matrix(cov, shrinkage);

    return {shrunk.selfadjointView<Eigen::Lower>(), mu};
}

void LedoitWolfCovarianceEstimator::translate_to_zero(DoubleMatrix& data, const DoubleVector& mu) {
    data.rowwise() -= mu.transpose();
}

void LedoitWolfCovarianceEstimator::translate_to_mu(DoubleMatrix& data, const DoubleVector& mu) {
    data.rowwise() += mu.transpose();
}

DoubleMatrix LedoitWolfCovarianceEstimator::shrunk_matrix(const DoubleMatrix& covariance, double shrinkage) {
    int d = covariance.cols();
    double trace_mean = covariance.trace() / d;
    DoubleMatrix shrunk = (1.0 - shrinkage) * covariance;
    for (int i = 0; i < d; ++i)
        shrunk(i, i) += shrinkage * trace_mean;
    return shrunk;
}