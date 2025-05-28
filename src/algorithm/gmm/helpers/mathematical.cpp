#include "mathematical.h"

#include <cmath>
#include <limits>

double log_sum_exp(const std::vector<double> &probabilities) {
    const double max_val = *std::max_element(probabilities.begin(), probabilities.end());

    if (std::isinf(max_val)) return max_val;

    double sum = 0.0;
    for (const double v: probabilities) {
        sum += std::exp(v - max_val);
    }

    return max_val + std::log(sum);
}

std::vector<std::vector<double> > estimate_weighted_log_probabilities(
    const Eigen::MatrixXd &data, const int k, const GMMResult &result,
    const std::vector<Eigen::MatrixXd> &precisionsCholesky
) {
    const int n = data.rows();
    const int d = data.cols();

    std::vector<double> log_det_cholesky(k, 0.0);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            log_det_cholesky[i] += log(precisionsCholesky[i](j, j));
        }
    }
    std::vector<std::vector<double> > log_probabilities(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i) {
        Eigen::VectorXd cluster = result.clusters.row(i).transpose();

        Eigen::MatrixXd y = (data * precisionsCholesky[i]).rowwise() - cluster.transpose() * precisionsCholesky[i];

        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int col = 0; col < y.cols(); ++col) {
                const double val = y(j, col);
                sum += val * val;
            }
            log_probabilities[j][i] = sum;
        }
    }
    std::vector<std::vector<double> > result_probabilities(n, std::vector<double>(k));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            result_probabilities[j][i] = -0.5 * (d * std::log(2 * M_PI) + log_probabilities[j][i]) +
                                         log_det_cholesky[i] + std::log(result.weights[i]);
        }
    }

    return result_probabilities;
}
