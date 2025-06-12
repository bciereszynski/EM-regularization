#include "CovarianceMatrixRegularizer.h"

Eigen::VectorXd CovarianceMatrixRegularizer::get_mu(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(weights.data(), weights.size());
    const double weight_sum = w.sum();

    Eigen::VectorXd mu = (data.array().colwise() * w.array()).colwise().sum() / weight_sum;

    return mu;
}
