#include "mathematical.h"

#include <cmath>
#include <limits>

double log_sum_exp(const std::vector<double> &probabilities) {
    double max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;

    for (const auto &p: probabilities) {
        if (std::isnan(p) || std::isnan(max)) {
            max = std::numeric_limits<double>::quiet_NaN();
            sum += std::exp(std::numeric_limits<double>::quiet_NaN());
        } else {
            if (p > max) {
                sum = (sum + one(sum)) * std::exp(max - p);
                max = p;
            } else if (p < max) {
                sum += std::exp(p - max);
            } else {
                sum += std::exp(zero(p - max));
            }
        }
    }

    return max + std::log1p(sum);
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
        Eigen::VectorXd cluster = Eigen::Map<const Eigen::VectorXd>(
            result.clusters[i].data(), result.clusters[i].size());

        Eigen::MatrixXd y = (data * precisionsCholesky[i]).rowwise() - (
                                cluster.transpose() * precisionsCholesky[i]); // n × d

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
