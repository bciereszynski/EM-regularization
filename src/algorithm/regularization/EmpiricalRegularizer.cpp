#include "EmpiricalRegularizer.h"
#include <numeric>

std::pair<DoubleMatrix, DoubleVector> EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const auto n = data.rows();
    const auto d = data.cols();

    Eigen::VectorXd mu = get_mu(data, weights);

    auto sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    DoubleMatrix covariance = Eigen::MatrixXd::Zero(d, d);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd diff = data.row(i).transpose() - mu;
        covariance += weights[i] * (diff * diff.transpose());
    }
    covariance /= sum_weights;

    return {covariance, mu};
}
