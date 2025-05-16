#include "SqMahalanobis.h"

SqMahalanobis::SqMahalanobis(const Eigen::MatrixXd &covariance, const bool skip_checks)
    : Q(covariance), checks_enabled(!skip_checks) {
    if (!checks_enabled) {
        llt_decomposition.compute(Q);
    } else {
        if (Q.rows() != Q.cols()) {
            throw std::invalid_argument("Covariance matrix must be square");
        }

        if (!(Q.transpose() - Q).isZero(1e-10)) {
            throw std::invalid_argument("Covariance matrix must be symmetric");
        }

        llt_decomposition.compute(Q);
    }
}

double SqMahalanobis::evaluate(const Eigen::VectorXd &x,
                               const Eigen::VectorXd &y) const {
    Eigen::VectorXd diff = x - y;

    if (checks_enabled) {
        if (diff.size() != Q.rows()) {
            throw std::invalid_argument("Vector dimensions must match covariance matrix dimensions");
        }
    }

    const double result = diff.transpose() * Q * diff;

    return result;
}
