#include "CovarianceMatrixRegularizer.h"
#include <numeric>

Eigen::VectorXd CovarianceMatrixRegularizer::get_mu(const DoubleMatrix &data,
                                                    const std::vector<double> &weights) {
    const int n = data.rows();
    const int d = data.cols();

    Eigen::VectorXd mu = Eigen::VectorXd::Zero(d);
    const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    for (int i = 0; i < n; ++i) {
        mu += weights[i] * data.row(i).transpose();
    }
    mu /= weight_sum;

    return mu;
}
