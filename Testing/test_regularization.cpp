#include <gtest/gtest.h>

#include "../src/algorithm/regularization/CovarianceMatrixRegularizer.h"
#include "../src/algorithm/regularization/EmpiricalRegularizer.h"

class RegularizationTest_FriendAccess : public ::testing::Test {
protected:
    static void TestGetMu_basic() {
        Eigen::MatrixXd data(3, 2);
        data << 1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0;

        const std::vector<double> weights = {0.2, 0.5, 0.3};

        Eigen::VectorXd expected_mu(2);
        expected_mu << 3.2, 4.2;

        const Eigen::VectorXd mu = CovarianceMatrixRegularizer::get_mu(data, weights);

        ASSERT_TRUE(mu.isApprox(expected_mu, 1e-6)) << "Mu does not match expected value.";
    }

    static void TestGetMu() {
        Eigen::MatrixXd X(11, 2);
        X << 8.446952398696654, 11.512125077077995,
                15.499610048784618, -3.161120404454193,
                -9.743336473406703, -18.80630870985824,
                -17.86342608516553, -11.382757975845214,
                14.398746744052609, 8.170480801024427,
                15.330418855520072, -8.057175056487976,
                4.882465698592702, -0.5870989988150339,
                -8.907657689495553, 18.466866205980494,
                2.492129816536738, -14.800565015825306,
                16.857415857407293, 5.338050934290413,
                -23.45789229394996, -2.679611133069019;

        const std::vector<double> weights = {
            1.0, 1.0, 4.944828537633115e-259, 9.1422604084118e-310, 1.0, 1.0, 1.0, 0.9921502699429642,
            1.3962696635361188e-53, 1.0, 0.0
        };

        Eigen::VectorXd expected_mu(2);
        expected_mu << 9.521802600275707, 4.510367687529266;

        const Eigen::VectorXd mu = CovarianceMatrixRegularizer::get_mu(X, weights);

        ASSERT_TRUE(mu.isApprox(expected_mu, 1e-6));
    }
};

TEST_F(RegularizationTest_FriendAccess, TestGetMu) {
    TestGetMu();
}

TEST_F(RegularizationTest_FriendAccess, TestGetMu_basic) {
    TestGetMu_basic();
}

TEST(EmpiricalRegularizerTest, Fit) {
    Eigen::MatrixXd X(11, 2);
    X << 8.446952398696654, 11.512125077077995,
            15.499610048784618, -3.161120404454193,
            -9.743336473406703, -18.80630870985824,
            -17.86342608516553, -11.382757975845214,
            14.398746744052609, 8.170480801024427,
            15.330418855520072, -8.057175056487976,
            4.882465698592702, -0.5870989988150339,
            -8.907657689495553, 18.466866205980494,
            2.492129816536738, -14.800565015825306,
            16.857415857407293, 5.338050934290413,
            -23.45789229394996, -2.679611133069019;

    const std::vector<double> weights = {
        1.0, 1.0, 4.944828537633115e-259, 9.1422604084118e-310, 1.0,
        1.0, 1.0, 0.9921502699429642, 1.3962696635361188e-53, 1.0, 0.0
    };

    Eigen::VectorXd expected_mu(2);
    expected_mu << 9.521802600275707, 4.510367687529266;
    Eigen::MatrixXd expected_covariance(2, 2);
    expected_covariance << 72.47093182603915, -47.76866097711419,
            -47.76866097711419, 71.38576144184982;

    EmpiricalRegularizer regularizer;
    auto [computed_covariance, computed_mu] = regularizer.fit(X, weights);

    ASSERT_TRUE(computed_mu.isApprox(expected_mu, 1e-6)) << "Mean vector (mu) does not match.";
    ASSERT_TRUE(computed_covariance.isApprox(expected_covariance, 1e-6)) << "Covariance matrix does not match.";
}
