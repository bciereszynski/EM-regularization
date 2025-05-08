#ifndef COVARIANCEMATRIXREGULARIZER_H
#define COVARIANCEMATRIXREGULARIZER_H

#include <vector>
#include <Eigen/Dense>

using DoubleMatrix = Eigen::MatrixXd;
using DoubleVector = Eigen::VectorXd;

class CovarianceMatrixRegularizer {
public:
  virtual ~CovarianceMatrixRegularizer() = default;

  virtual std::pair<DoubleMatrix, DoubleVector> fit(
    const DoubleMatrix &data, const std::vector<double> &weights) = 0;

protected:
  static DoubleVector get_mu(const DoubleMatrix &data, const std::vector<double> &weights);
};

#endif //COVARIANCEMATRIXREGULARIZER_H
