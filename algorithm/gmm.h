#ifndef GMM_H
#define GMM_H

#include <random>

constexpr bool DEFAULT_VERBOSE = false;
constexpr double DEFAULT_TOLERANCE = 1e-6;
constexpr int DEFAULT_MAX_ITERATIONS = 100;

// The GMM is a clustering algorithm that models the underlying data distribution as a mixture of Gaussian distributions.
class GMM {
public:
    double tolerance;
    int max_iterations;
    bool verbose;
    std::mt19937 rng;
    bool decompose_if_fails;

    explicit GMM(
        double tolerance = DEFAULT_TOLERANCE,
        int max_iterations = DEFAULT_MAX_ITERATIONS,
        bool verbose = DEFAULT_VERBOSE,
        unsigned int seed = std::random_device{}(),
        bool decompose_if_fails = true
    );

    void fit();
};

#endif // GMM_H
