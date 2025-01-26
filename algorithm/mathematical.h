#ifndef MATHEMATICAL_H
#define MATHEMATICAL_H

#include <vector>

template<typename T>
T one(const T &x) {
    return static_cast<T>(1);
}

template<typename T>
T zero(const T &x) {
    return static_cast<T>(0);
}

double log_sum_exp(const std::vector<double> &probabilities);

#endif //MATHEMATICAL_H
