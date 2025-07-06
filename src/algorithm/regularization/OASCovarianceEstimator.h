#ifndef OASESTIMATOR_H
#define OASESTIMATOR_H

#include "CovarianceMatrixRegularizer.h"
#include "EmpiricalRegularizer.h"

class OASCovarianceEstimator : public EmpiricalRegularizer {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const std::vector<double> &weights) override;

private:
    void translate_to_zero(DoubleMatrix& data, const DoubleVector& mu);
    void translate_to_mu(DoubleMatrix& data, const DoubleVector& mu);
    std::pair<DoubleMatrix, DoubleVector> empirical(const DoubleMatrix& data, const std::vector<double>& weights);
    DoubleMatrix shrunk_matrix(const DoubleMatrix& covariance, double shrinkage);
};

#endif // OASESTIMATOR_H