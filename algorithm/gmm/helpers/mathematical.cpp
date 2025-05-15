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
