#include "EmpiricalRegularizer.h"
#include <numeric>

std::pair<DoubleMatrix, std::vector<double> > EmpiricalRegularizer::fit(
    const DoubleMatrix &data, const std::vector<double> &weights) {
    const int n = data.size();
    const int d = data[0].size();

    std::vector<double> mu = get_mu(data, weights);

    DoubleMatrix translated(n, std::vector<double>(d));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            translated[i][j] = data[i][j] - mu[j];
        }
    }

    DoubleMatrix covariance(d, std::vector<double>(d, 0.0));
    const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < d; ++k) {
                covariance[j][k] += weights[i] * translated[i][j] * translated[i][k] / weight_sum;
            }
        }
    }

    return {covariance, mu};
}
