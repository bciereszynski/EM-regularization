#ifndef OASESTIMATOR_H
#define OASESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "LedoitWolfCovarianceEstimator.h"

class OASCovarianceEstimator : public LedoitWolfCovarianceEstimator {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const DoubleVector &weights) override;
};

#endif // OASESTIMATOR_H
