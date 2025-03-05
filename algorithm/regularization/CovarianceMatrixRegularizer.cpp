#include "CovarianceMatrixRegularizer.h"
#include <numeric>

std::vector<double> CovarianceMatrixRegularizer::get_mu(const DoubleMatrix &data,
                                                        const std::vector<double> &weights) {
    const int n = data.size();
    const int d = data[0].size();

    const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);

    std::vector<double> mu(d, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            mu[j] += data[i][j] * weights[i] / weight_sum;
        }
    }

    return mu;
}
