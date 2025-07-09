#include <gtest/gtest.h>

#include "../src/algorithm/regularization/CovarianceMatrixRegularizer.h"
#include "../src/algorithm/regularization/EmpiricalRegularizer.h"
#include "../src/algorithm/regularization/ShrunkCovarianceEstimator.h"

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

class EmpiricalRegularizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        X_ << 8.446952398696654, 11.512125077077995,
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
    }

    Eigen::MatrixXd X_{11, 2};
};

TEST_F(RegularizationTest_FriendAccess, TestGetMu) {
    TestGetMu();
}

TEST_F(RegularizationTest_FriendAccess, TestGetMu_basic) {
    TestGetMu_basic();
}

TEST_F(EmpiricalRegularizerTest, Fit) {
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
    auto [computed_covariance, computed_mu] = regularizer.fit(X_, weights);

    ASSERT_TRUE(computed_mu.isApprox(expected_mu, 1e-6)) << "Mean vector (mu) does not match.";
    ASSERT_TRUE(computed_covariance.isApprox(expected_covariance, 1e-6)) << "Covariance matrix does not match.";
}

TEST_F(EmpiricalRegularizerTest, ExtremeWeights) {
    const std::vector<double> weights = {
        1.0, 1.0, 1e-259, 1e-310, 1.0,
        1.0, 1.0, 0.99215, 1e-53, 1.0, 0.0
    };

    EmpiricalRegularizer regularizer;
    auto [cov, mu] = regularizer.fit(X_, weights);

    ASSERT_FALSE(mu.hasNaN());
    ASSERT_FALSE(cov.hasNaN());
    ASSERT_TRUE(cov.llt().info() == Eigen::Success); // Positive definite
}

TEST(ShrunkCovarianceTest, ExactNumericalComparison) {
    // Input data
    Eigen::MatrixXd X(11, 2);
    X << 8.44695, 11.5121,
            15.4996, -3.16112,
            -9.74334, -18.8063,
            -17.8634, -11.3828,
            14.3987, 8.17048,
            15.3304, -8.05718,
            4.88247, -0.587099,
            -8.90766, 18.4669,
            2.49213, -14.8006,
            16.8574, 5.33805,
            -23.4579, -2.67961;

    const std::vector<double> weights = {
        0.061250745076178954, 0.999999999999841, 0.0016794498296731847,
        0.00022390588318295702, 0.01695985184643251, 1.0,
        1.0, 0.9999999997194706, 0.9999999763559252,
        0.025782471947576716, 0.00030634259915663547
    };

    constexpr double tolerance = 1e-4;

    Eigen::Vector2d expected_mu;
    expected_mu << 5.966395670856878, -1.4086244814822022;

    Eigen::Matrix2d expected_empirical_cov;
    expected_empirical_cov << 81.9587, -63.2966,
            -63.2966, 124.518;

    Eigen::Matrix2d expected_shrunk_cov;
    expected_shrunk_cov << 84.0867, -56.967,
            -56.967, 122.39;

    ShrunkCovarianceEstimator estimator;
    auto [computed_cov, computed_mu] = estimator.fit(X, weights);

    EXPECT_TRUE(computed_mu.isApprox(expected_mu, tolerance))
        << "Mu mismatch:\nComputed:\n" << computed_mu
        << "\nExpected:\n" << expected_mu;

    const Eigen::Matrix2d computed_sym = 0.5 * (computed_cov + computed_cov.transpose());

    EXPECT_TRUE(computed_sym.isApprox(expected_shrunk_cov, tolerance))
        << "Covariance mismatch:\nComputed:\n" << computed_sym
        << "\nExpected:\n" << expected_shrunk_cov;

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(computed_sym);
    Eigen::Vector2d expected_eigenvalues;
    expected_eigenvalues << 43.1383, 163.3386;

    EXPECT_TRUE(solver.eigenvalues().isApprox(expected_eigenvalues, tolerance))
        << "Eigenvalues mismatch:\nComputed:\n" << solver.eigenvalues()
        << "\nExpected:\n" << expected_eigenvalues;
}

