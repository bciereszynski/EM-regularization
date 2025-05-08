#include "EmpiricalRegularizer.h"
#include <numeric>

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const auto n = data.rows();

    Eigen::VectorXd mu = get_mu(data, weights);

    DoubleMatrix translated = data.rowwise() - mu.transpose();
    DoubleMatrix covariance = (translated.transpose() * translated) / n;

    return {covariance, mu};
}
