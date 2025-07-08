#ifndef OASESTIMATOR_H
#define OASESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "LedoitWolfCovarianceEstimator.h"

class OASCovarianceEstimator : public LedoitWolfCovarianceEstimator {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const std::vector<double> &weights) override;

private:
    std::pair<DoubleMatrix, DoubleVector> empirical(const DoubleMatrix &data, const std::vector<double> &weights);
};

#endif // OASESTIMATOR_H
