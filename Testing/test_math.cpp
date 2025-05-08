#include "../algorithm/gmm/gmm.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

class GMMTest_FriendAccess : public ::testing::Test {
protected:
    GMM gmm{1e-3, 100, false, 42, false};

    void TestComputePrecisionCholesky() const {
        constexpr int k = 3;
        constexpr int d = 2;
        GMMResult result(d, 10, k);
        for (int i = 0; i < k; ++i) {
            result.covariances[i] = Eigen::MatrixXd::Identity(d, d);
        }
        std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd(d, d));
        gmm.compute_precision_cholesky(result, precisions_cholesky);
        const Eigen::MatrixXd expected = Eigen::MatrixXd::Identity(d, d);

        for (int i = 0; i < k; ++i) {
            ASSERT_TRUE(precisions_cholesky[i].isApprox(expected, 1e-6))
                << "Precision matrix " << i << " does not match expected identity.";
        }
    }

    void TestComputePrecisionCholesky_NonIdentityCovariances() const {
        constexpr int k = 3;
        constexpr int d = 2;
        GMMResult result(d, 10, k);

        result.covariances[0] << 33.8893487690863, 32.65001571140097,
                32.65001571140097, 47.0946663276457;

        result.covariances[1] << 102.29447238814718, 39.51710506986864,
                39.51710506986864, 60.53416336025682;

        result.covariances[2] << 73.45087666482745, 30.54605547510448,
                30.54605547510448, 54.97462858435647;

        std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd(d, d));
        gmm.compute_precision_cholesky(result, precisions_cholesky);

        std::vector<Eigen::MatrixXd> expected = {
            (Eigen::MatrixXd(2, 2) << 0.17177833476190066, 0.0,
             -0.24362417677019974, 0.25287169133777315).finished(),

            (Eigen::MatrixXd(2, 2) << 0.09887213601054862, 0.0,
             -0.057416313809908096, 0.14862858798164388).finished(),

            (Eigen::MatrixXd(2, 2) << 0.11668136690747105, 0.0,
             -0.06396386718928732, 0.15380716255671226).finished()
        };

        for (int i = 0; i < k; ++i) {
            ASSERT_TRUE(precisions_cholesky[i].isApprox(expected[i], 1e-6))
                << "Precision matrix " << i << " does not match expected output.\n"
                << "Expected:\n" << expected[i] << "\nGot:\n" << precisions_cholesky[i];
        }
    }
};

// Test case
TEST_F(GMMTest_FriendAccess, ComputePrecisionCholesky_IdentityInput_YieldsIdentityOutput) {
    TestComputePrecisionCholesky();
}

TEST_F(GMMTest_FriendAccess, ComputePrecisionCholesky_RealisticCovariances_YieldsExpectedOutput) {
    TestComputePrecisionCholesky_NonIdentityCovariances();
}
