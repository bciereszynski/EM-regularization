#include "LedoitWolfCovarianceEstimator.h"

std::pair<DoubleMatrix, DoubleVector> LedoitWolfCovarianceEstimator::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
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
    const double shrinkage = (beta <= 0.0) ? 0.0 : beta / delta;

    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), weights.size());
    const double sum_weights = w.sum();
    DoubleMatrix weighted_centered = X.array().colwise() * w.array();
    const DoubleMatrix covariance = (weighted_centered.transpose() * X) / sum_weights;

    DoubleMatrix shrunk = shrunk_matrix(covariance, shrinkage);

    return {shrunk, mu};
}

void LedoitWolfCovarianceEstimator::translate_to_zero(DoubleMatrix &data, const DoubleVector &mu) {
    data.rowwise() -= mu.transpose();
}

void LedoitWolfCovarianceEstimator::translate_to_mu(DoubleMatrix &data, const DoubleVector &mu) {
    data.rowwise() += mu.transpose();
}
