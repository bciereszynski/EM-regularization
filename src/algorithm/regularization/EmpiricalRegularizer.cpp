#include "EmpiricalRegularizer.h"
#include <numeric>

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.rows();

    Eigen::VectorXd mu = get_mu(data, weights);

    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), n);
    const double sum_weights = w.sum();

    DoubleMatrix centered = data.rowwise() - mu.transpose();
    DoubleMatrix weighted_centered = centered.array().colwise() * w.array();
    DoubleMatrix covariance = (weighted_centered.transpose() * centered) / sum_weights;

    return {covariance, mu};
}
