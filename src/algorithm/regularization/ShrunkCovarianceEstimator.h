#ifndef SHRUNKCOVARIANCEESTIMATOR_H
#define SHRUNKCOVARIANCEESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "EmpiricalRegularizer.h"

#define DEFAULT_SHRINKAGE 0.1

class ShrunkCovarianceEstimator : public EmpiricalRegularizer {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const std::vector<double> &weights) override;

protected:
    static std::pair<DoubleMatrix, DoubleVector> shrunk(const DoubleMatrix &data, const std::vector<double> &weights,
                                                        double shrinkage);

    static DoubleMatrix shrunk_matrix(const DoubleMatrix &covariance, double shrinkage);
};

#endif // SHRUNKCOVARIANCEESTIMATOR_H
