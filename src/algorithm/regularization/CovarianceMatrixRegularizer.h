#ifndef COVARIANCEMATRIXREGULARIZER_H
#define COVARIANCEMATRIXREGULARIZER_H

#include <vector>
#include <Eigen/Dense>

#define EPS 1e-6

using DoubleMatrix = Eigen::MatrixXd;
using DoubleVector = Eigen::VectorXd;

class CovarianceMatrixRegularizer {
public:
  virtual ~CovarianceMatrixRegularizer() = default;

  virtual std::pair<DoubleMatrix, DoubleVector> fit(
    const DoubleMatrix &data, const DoubleVector &weights) = 0;

protected:
  static DoubleVector get_mu(const DoubleMatrix &data, const DoubleVector &weights);

  friend class RegularizationTest_FriendAccess;
};

#endif //COVARIANCEMATRIXREGULARIZER_H
