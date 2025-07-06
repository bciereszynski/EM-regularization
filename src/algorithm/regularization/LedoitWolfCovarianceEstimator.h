#ifndef LEDOITWOLFESTIMATOR_H
#define LEDOITWOLFESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "EmpiricalRegularizer.h"

class LedoitWolfCovarianceEstimator : public EmpiricalRegularizer {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const std::vector<double> &weights) override;

private:
    void translate_to_zero(DoubleMatrix& data, const DoubleVector& mu);
    void translate_to_mu(DoubleMatrix& data, const DoubleVector& mu);
    DoubleMatrix shrunk_matrix(const DoubleMatrix& covariance, double shrinkage);
};

#endif // LEDOITWOLFESTIMATOR_H