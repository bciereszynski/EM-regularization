#ifndef EMPIRICALREGULARIZER_H
#define EMPIRICALREGULARIZER_H

#include "CovarianceMatrixRegularizer.h"


class EmpiricalRegularizer : public CovarianceMatrixRegularizer {
public:
    std::pair<DoubleMatrix, DoubleVector> fit(
        const DoubleMatrix &data, const DoubleVector &weights) override;

protected:
    static std::pair<DoubleMatrix, DoubleVector> compute_empirical(
        const DoubleMatrix &data,
        const DoubleVector &weights);
};

#endif //EMPIRICALREGULARIZER_H
