#include "CovarianceMatrixRegularizer.h"

Eigen::VectorXd CovarianceMatrixRegularizer::get_mu(
    const DoubleMatrix &data, const DoubleVector &weights) {
    const double weight_sum = weights.sum();

    Eigen::VectorXd mu = (data.array().colwise() * weights.array()).colwise().sum() / weight_sum;

    return mu;
}
