#ifndef GMMRESULT_H
#define GMMRESULT_H

#include <vector>
#include <Eigen/Dense>

class GMMResult {
public:
    std::vector<int> assignments;
    std::vector<double> weights;
    std::vector<std::vector<double> > clusters;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > covariances;
    double objective;
    int iterations;
    double elapsed;
    bool converged;
    int k;

    GMMResult(
        const std::vector<int> &assignments,
        const std::vector<double> &weights,
        const std::vector<std::vector<double> > &clusters,
        const std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > &covariances,
        const double objective = -std::numeric_limits<double>::infinity(),
        const int iterations = 0,
        const double elapsed = 0.0,
        const bool converged = false
    ) : assignments(assignments),
        weights(weights),
        clusters(clusters),
        covariances(covariances),
        objective(objective),
        iterations(iterations),
        elapsed(elapsed),
        converged(converged),
        k(static_cast<int>(clusters.size())) {
    }

    GMMResult(const int d, const int n, const int k)
        : GMMResult(
            std::vector<int>(n, 0),
            std::vector<double>(k, 1.0 / k),
            std::vector<std::vector<double> >(k, std::vector<double>(d, 0)),
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(k,
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(d, d))
        ) {
    }

    void reset() {
        objective = -std::numeric_limits<double>::infinity();
    };
};

inline bool is_better(const GMMResult &a, const GMMResult &b) {
    return a.objective > b.objective;
}

#endif // GMMRESULT_H
