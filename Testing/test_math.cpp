#include "../src/algorithm/gmm/gmm.h"
#include <gtest/gtest.h>
#include "../src/data.h"
#include <Eigen/Dense>
#include "../src/algorithm/gmm/helpers/mathematical.h"


class MathTest_FriendAccess : public ::testing::Test {
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
        gmm.compute_precisions_cholesky(result, precisions_cholesky);
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
        gmm.compute_precisions_cholesky(result, precisions_cholesky);

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
TEST_F(MathTest_FriendAccess, ComputePrecisionCholesky_IdentityInput_YieldsIdentityOutput) {
    TestComputePrecisionCholesky();
}

TEST_F(MathTest_FriendAccess, ComputePrecisionCholesky_RealisticCovariances_YieldsExpectedOutput) {
    TestComputePrecisionCholesky_NonIdentityCovariances();
}

TEST(MathTest, EstimateWeightedLogProbabilities_SimpleCase) {
    constexpr int n = 2;
    constexpr int d = 2;
    constexpr int k = 1;

    Eigen::MatrixXd data(n, d);
    data << 1.0, 2.0,
            3.0, 4.0;

    GMMResult result(d, n, k);
    result.clusters.row(0) = Eigen::VectorXd::Constant(2, 0.0);
    result.weights[0] = 1.0;

    std::vector<Eigen::MatrixXd> precisions_cholesky(1, Eigen::MatrixXd::Identity(d, d));

    const std::vector<std::vector<double> > log_probs = estimate_weighted_log_probabilities(
        data, k, result, precisions_cholesky);

    ASSERT_EQ(log_probs.size(), 2);
    ASSERT_EQ(log_probs[0].size(), 1);

    const double log2pi = std::log(2 * M_PI);
    const double expected0 = -0.5 * (2 * log2pi + 1.0 * 1.0 + 2.0 * 2.0); // ||[1,2]||^2 = 1 + 4 = 5
    const double expected1 = -0.5 * (2 * log2pi + 3.0 * 3.0 + 4.0 * 4.0); // ||[3,4]||^2 = 9 + 16 = 25

    EXPECT_NEAR(log_probs[0][0], expected0, 1e-6);
    EXPECT_NEAR(log_probs[1][0], expected1, 1e-6);
}

TEST(MathTest, EstimateWeightedLogProbabilities_RealData) {
    int k = 3;
    constexpr int d = 2;
    std::vector<int> expected_clusters;
    std::vector<std::vector<double> > data_raw = load_data_from_file(
        "../test_data/3_2_-0.26_1.csv", expected_clusters, k);
    Eigen::MatrixXd data(data_raw.size(), data_raw[0].size());
    for (size_t i = 0; i < data_raw.size(); ++i) {
        data.row(i) = Eigen::VectorXd::Map(data_raw[i].data(), data_raw[i].size());
    }
    const int n = data.rows();

    GMMResult result(d, n, k);
    result.clusters << 10.706315370566315, -15.655038396575486,
            1.5346346971608753, 10.172773119928657,
            0.8125445826316164, -9.318465733509676;
    result.weights = {0.12309068557448444, 0.30756460120925033, 0.5693447132162652};

    std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd(d, d));

    precisions_cholesky[0] << 0.17177833476190066, -0.24362417677019974,
            0.0, 0.25287169133777315;

    precisions_cholesky[1] << 0.09887213601054862, -0.057416313809908096,
            0.0, 0.14862858798164388;

    precisions_cholesky[2] << 0.11668136690747105, -0.06396386718928732,
            0.0, 0.15380716255671226;

    const std::vector<std::vector<double> > log_probs = estimate_weighted_log_probabilities(
        data, k, result, precisions_cholesky);

    ASSERT_EQ(log_probs.size(), n);
    ASSERT_EQ(log_probs[0].size(), k);

    const std::vector<std::vector<double> > expected = {
        {-34.67444478856755, -7.490287069833688, -10.505420842688025},
        {-9.391341909521708, -12.06466757490316, -7.889934866592154},
        {-21.996834185639685, -14.555142916332544, -7.487430481051719},
        {-51.43748366931587, -11.260459416036612, -9.180469588500568},
        {-20.404300474376164, -8.582910721767998, -9.335866619446161}
    };

    for (int i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            EXPECT_NEAR(log_probs[i][j], expected[i][j], 1e-5)
                    << "Mismatch at row " << i << ", col " << j;
        }
    }
}

