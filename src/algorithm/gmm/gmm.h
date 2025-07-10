#ifndef GMM_H
#define GMM_H

#include <random>
#include "gmmResult.h"
#include "../regularization/EmpiricalRegularizer.h"

constexpr bool DEFAULT_VERBOSE = false;
constexpr double DEFAULT_TOLERANCE = 1e-3;
constexpr int DEFAULT_MAX_ITERATIONS = 200;
inline auto DEFAULT_REGULARIZER = EmpiricalRegularizer();

// The GMM is a clustering algorithm that models the underlying data distribution as a mixture of Gaussian distributions.
class GMM {
    double tolerance;
    int max_iterations;
    bool verbose;
    std::mt19937 rng;
    CovarianceMatrixRegularizer *regularizer;

public:
    explicit GMM(
        double tolerance = DEFAULT_TOLERANCE,
        int max_iterations = DEFAULT_MAX_ITERATIONS,
        bool verbose = DEFAULT_VERBOSE,
        unsigned int seed = std::random_device{}(),
        CovarianceMatrixRegularizer *regularizer = &DEFAULT_REGULARIZER

    );

    explicit GMM(
        const std::mt19937 &rng,
        CovarianceMatrixRegularizer *regularizer,
        double tolerance = DEFAULT_TOLERANCE,
        int max_iterations = DEFAULT_MAX_ITERATIONS,
        bool verbose = DEFAULT_VERBOSE
    );

    GMMResult fit(const Eigen::MatrixXd &data, GMMResult &result) const;

    GMMResult fit(const Eigen::MatrixXd &data, int k);

    GMMResult fit(const std::vector<std::vector<double> > &data, int k);

    [[nodiscard]] GMMResult fit(const Eigen::MatrixXd &data,
                                const std::vector<int> &initial_clusters);

private:
    static void compute_precisions_cholesky(GMMResult &result,
                                            std::vector<Eigen::MatrixXd> &precisions_cholesky);

    std::tuple<std::vector<double>, Eigen::MatrixXd,
        std::vector<Eigen::MatrixXd> >
    estimate_gaussian_parameters(const Eigen::MatrixXd &, int,
                                 Eigen::MatrixXd &) const;

    static std::pair<double, Eigen::MatrixXd> expectation_step(
        const Eigen::MatrixXd &data,
        int k,
        const GMMResult &result,
        const std::vector<Eigen::MatrixXd> &precisionsCholesky);

    void maximization_step(const Eigen::MatrixXd &data, int k, GMMResult &result,
                           const Eigen::MatrixXd &log_responsibilities,
                           std::vector<Eigen::MatrixXd> &precision_cholesky) const;

    friend class GMMTest_FriendAccess;
    friend class MathTest_FriendAccess;
};

#endif // GMM_H
