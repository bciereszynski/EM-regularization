#ifndef EMPIRICALREGULARIZER_H
#define EMPIRICALREGULARIZER_H

#include "CovarianceMatrixRegularizer.h"


class EmpiricalRegularizer : public CovarianceMatrixRegularizer {
public:
    std::pair<DoubleMatrix, std::vector<double> > fit(
        const DoubleMatrix &data, const std::vector<double> &weights) override;
};


#endif //EMPIRICALREGULARIZER_H
