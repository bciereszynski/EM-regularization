#include "LedoitWolfCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> LedoitWolfCovarianceEstimator::fit(
    const DoubleMatrix &data, const DoubleVector &weights) {
    const int n = data.rows();
    const int d = data.cols();

    DoubleVector mu = get_mu(data, weights);

    DoubleMatrix X = data.rowwise() - mu.transpose();

    DoubleMatrix X2 = X.array().square();
    const DoubleVector trace = X2.colwise().sum() / n;
    const double trace_mean = trace.sum() / d;

    const double beta_coeff = (X2.transpose() * X2).sum();

    DoubleMatrix XtX = X.transpose() * X;
    const double delta_coeff = XtX.array().square().sum() / (n * n);

    double beta = 1.0 / (n * d) * (beta_coeff / n - delta_coeff);
    double delta = delta_coeff - 2.0 * trace_mean * trace.sum() + d * trace_mean * trace_mean;
    delta /= d;

    beta = std::min(beta, delta);
    const double shrinkage = (abs(beta) <= EPS) ? 0.0 : beta / delta;

    return shrunk(data, weights, shrinkage);
}

void LedoitWolfCovarianceEstimator::translate_to_zero(DoubleMatrix &data, const DoubleVector &mu) {
    data.rowwise() -= mu.transpose();
}

void LedoitWolfCovarianceEstimator::translate_to_mu(DoubleMatrix &data, const DoubleVector &mu) {
    data.rowwise() += mu.transpose();
}
