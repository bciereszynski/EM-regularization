#ifndef SQ_MAHALANOBIS_H
#define SQ_MAHALANOBIS_H

#include <Eigen/Dense>
#include <stdexcept>

class SqMahalanobis {
private:
    Eigen::MatrixXd Q;

    Eigen::LLT<Eigen::MatrixXd> llt_decomposition;
    bool checks_enabled;

public:
    explicit SqMahalanobis(const Eigen::MatrixXd &covariance, bool skip_checks = true);

    [[nodiscard]] double evaluate(const Eigen::VectorXd &x,
                                  const Eigen::VectorXd &y) const;
};

namespace Distances {
    template<typename DistanceMetric>
    double evaluate(const DistanceMetric &dist,
                    const Eigen::VectorXd &x,
                    const Eigen::VectorXd &y) {
        return dist.evaluate(x, y);
    }
}

#endif // SQ_MAHALANOBIS_H
