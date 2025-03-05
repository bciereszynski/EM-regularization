#ifndef COVARIANCEMATRIXREGULARIZER_H
#define COVARIANCEMATRIXREGULARIZER_H

#include <vector>

using DoubleMatrix = std::vector<std::vector<double> >;


class CovarianceMatrixRegularizer {
public:
  virtual ~CovarianceMatrixRegularizer() = default;

  virtual std::pair<DoubleMatrix, std::vector<double> > fit(
    const DoubleMatrix &data, const std::vector<double> &weights) = 0;

protected:
  static std::vector<double> get_mu(const DoubleMatrix &data, const std::vector<double> &weights);
};

#endif //COVARIANCEMATRIXREGULARIZER_H
