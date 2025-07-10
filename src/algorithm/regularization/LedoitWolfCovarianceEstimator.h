#ifndef LEDOITWOLFESTIMATOR_H
#define LEDOITWOLFESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "ShrunkCovarianceEstimator.h"

class LedoitWolfCovarianceEstimator : public ShrunkCovarianceEstimator {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const DoubleVector &weights) override;

protected:
    static void translate_to_zero(DoubleMatrix &data, const DoubleVector &mu);

    static void translate_to_mu(DoubleMatrix &data, const DoubleVector &mu);
};

#endif // LEDOITWOLFESTIMATOR_H
