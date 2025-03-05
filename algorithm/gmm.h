#ifndef GMM_H
#define GMM_H

#include <random>
#include "gmmResult.h"

constexpr bool DEFAULT_VERBOSE = true;
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

    GMMResult fit(const std::vector<std::vector<double> > &data, GMMResult &result);

    GMMResult fit(const std::vector<std::vector<double> > &data, int k);

    GMMResult fit(const std::vector<std::vector<double> > &data, const std::vector<int> &initial_clusters);

private:
    std::pair<double, std::vector<std::vector<double> > > expectation(
        const std::vector<std::vector<double> > &data, int k, const GMMResult &result,
        const std::vector<Eigen::MatrixXd> &precisionsCholesky);

    void maximization_step(const std::vector<std::vector<double> > &data, int k, GMMResult &result,
                           const std::vector<std::vector<double> > &log_responsibilities,
                           std::vector<Eigen::MatrixXd> &precision_cholesky);


    std::tuple<std::vector<double>, std::vector<std::vector<double> >, std::vector<Eigen::Matrix<double, Eigen::Dynamic,
        Eigen::Dynamic> > >
    estimate_gaussian_parameters(
        const std::vector<std::vector<double> > &data,
        int k,
        std::vector<std::vector<double> > &responsibilities);

    void computePrecisionCholesky(GMMResult &result,
                                  std::vector<Eigen::MatrixXd> &precisions_cholesky);

    static std::vector<std::vector<double> > estimateWeightedLogProbabilities(
        const std::vector<std::vector<double> > &data, int k, const GMMResult &result,
        const std::vector<Eigen::MatrixXd> &precisionsCholesky
    );
};

#endif // GMM_H
