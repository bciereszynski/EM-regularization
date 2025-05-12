#include <gtest/gtest.h>
#include "../src/algorithm/gmm/gmm.h"
#include "../src/data.h"
#include <Eigen/Dense>

TEST(GMMTest, BasicClustering) {
    Eigen::MatrixXd data(6, 1);
    data << 1.0, 1.1, 0.9, 5.0, 5.1, 4.9;

    GMM model(1e-4, 100, false, 42, true);
    const auto result = model.fit(data, 2);

    EXPECT_EQ(result.assignments.size(), 6);
    EXPECT_EQ(result.weights.size(), 2);
    EXPECT_EQ(result.clusters.size(), 2);
    EXPECT_EQ(result.covariances.size(), 2);

    for (const auto &cov: result.covariances) {
        EXPECT_GT(cov(0, 0), 0);
    }
}

TEST(GMMTest, CorrectAssignments) {
    const std::vector<std::vector<double> > data = {
        {0.0, 0.1}, {0.1, -0.1}, {-0.1, 0.2}, // Cluster 0
        {5.0, 5.1}, {5.1, 4.9}, {4.9, 5.2} // Cluster 1
    };

    GMM model(1e-6, 100, false, 42, true);

    const auto result = model.fit(data, 2);

    int count_cluster0 = 0;
    int count_cluster1 = 0;
    for (int i = 0; i < 3; ++i) {
        if (result.assignments[i] == result.assignments[0]) {
            count_cluster0++;
        }
    }
    for (int i = 3; i < 6; ++i) {
        if (result.assignments[i] == result.assignments[3]) {
            count_cluster1++;
        }
    }

    EXPECT_EQ(count_cluster0, 3);
    EXPECT_EQ(count_cluster1, 3);
}

TEST(GMMTest, WeightsSumToOne) {
    GMM model(1e-6, 100, false, 42, true);
    Eigen::MatrixXd data(4, 2);
    data << 0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1;
    const auto result = model.fit(data, 2);
    const double sum = std::accumulate(result.weights.begin(), result.weights.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-8);
}

TEST(GMMTest, HandlesEmptyInput) {
    GMM model(1e-6, 100, false, 42, true);
    const Eigen::MatrixXd data;
    EXPECT_NO_THROW({
        const auto result = model.fit(data, 2);
        EXPECT_EQ(result.assignments.size(), 0);
        });
}

TEST(GMMTest, HandlesMinimalInputOnePoint) {
    GMM model(1e-6, 100, false, 42, true);
    Eigen::MatrixXd data(1, 2); // One point in 2D
    data << 1.0, 2.0;
    EXPECT_NO_THROW({
        const auto result = model.fit(data, 1);
        EXPECT_EQ(result.assignments.size(), 1);
        EXPECT_EQ(result.assignments[0], 0);
        });
}

TEST(GMMTest, HandlesHighDimensionalData) {
    GMM model(1e-6, 100, false, 42, true);
    const Eigen::MatrixXd data = Eigen::MatrixXd::Random(10, 50);
    EXPECT_NO_THROW({
        const auto result = model.fit(data, 3);
        EXPECT_EQ(result.assignments.size(), 10);
        });
}

class GMMTest_FriendAccess : public ::testing::Test {
protected:
    GMM gmm{1e-3, 100, false, 42, false};

    static void TestExpectationStep() {
        int k = 3;
        std::vector<int> expected_clusters;
        // TODO review all matrixes to EIGEN
        std::vector<std::vector<double> > data_raw = load_data_from_file(
            "../test_data/3_2_-0.26_1.csv", expected_clusters, k);
        Eigen::MatrixXd data(data_raw.size(), data_raw[0].size());
        for (size_t i = 0; i < data_raw.size(); ++i) {
            data.row(i) = Eigen::VectorXd::Map(data_raw[i].data(), data_raw[i].size());
        }
        const int n = data.rows();
        const int d = data.cols();

        GMMResult result(d, n, k);
        result.clusters[0] = {8.072821980338327, -18.486899691951532};
        result.clusters[1] = {-8.15593365254254, 15.568038996903327};
        result.clusters[2] = {3.2971078319133493, -15.064491056854397};
        result.weights = {0.3333333333333333, 0.3333333333333333, 0.3333333333333333};

        std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd::Identity(d, d));

        const auto [mean_log_likelihood, log_responsibilities] =
                GMM::expectation_step(data, k, result, precisions_cholesky);

        EXPECT_NEAR(mean_log_likelihood, -100.48916207565158, 1e-6);

        const std::vector<std::vector<double> > expected_log_responsibilities = {
            {-303.987598853661, 0.0, -220.36558061847856},
            {-0.5640781381025022, -310.7288120901821, -0.8413788673385341},
            {-66.73156857059266, -500.03062047627327, 0.0},
            {-130.91721955111578, -179.62875633005083, 0.0},
            {-93.5978919047501, 0.0, -49.83641435898852}
        };

        for (size_t i = 0; i < expected_log_responsibilities.size(); ++i) {
            for (size_t j = 0; j < k; ++j) {
                EXPECT_NEAR(log_responsibilities(i, j), expected_log_responsibilities[i][j], 1e-6)
                    << "Mismatch at row " << i << ", col " << j;
            }
        }
    }

    static void TestExpectationStep_SecondCase() {
        int k = 3;
        std::vector<int> expected_clusters;
        std::vector<std::vector<double> > data_raw = load_data_from_file(
            "../test_data/3_2_-0.26_1.csv", expected_clusters, k);
        Eigen::MatrixXd data(data_raw.size(), data_raw[0].size());
        for (size_t i = 0; i < data_raw.size(); ++i) {
            data.row(i) = Eigen::VectorXd::Map(data_raw[i].data(), data_raw[i].size());
        }
        const int n = data.rows();
        const int d = data.cols();

        GMMResult result(d, n, k);
        result.clusters[0] = {10.706315370566315, -15.655038396575486};
        result.clusters[1] = {1.5346346971608753, 10.172773119928657};
        result.clusters[2] = {0.8125445826316164, -9.318465733509676};
        result.weights = {0.12309068557448444, 0.30756460120925033, 0.5693447132162652};

        std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd(d, d));
        precisions_cholesky[0] << 0.17177833476190066, -0.24362417677019974,
                0.0, 0.25287169133777315;
        precisions_cholesky[1] << 0.09887213601054862, -0.057416313809908096,
                0.0, 0.14862858798164388;
        precisions_cholesky[2] << 0.11668136690747105, -0.06396386718928732,
                0.0, 0.15380716255671226;

        const auto [mean_log_likelihood, log_responsibilities] =
                GMM::expectation_step(data, k, result, precisions_cholesky);

        EXPECT_NEAR(mean_log_likelihood, -7.5749208453554004, 1e-6);

        const std::vector<std::vector<double> > expected_log_responsibilities = {
            {-27.232032487796047, -0.04787476906218746, -3.063008541916525},
            {-1.715062305720429, -4.3883879711018805, -0.21365526279087454},
            {-14.510256021188855, -7.068564751881714, -0.0008523166008886918},
            {-42.37473621066418, -2.1977119573849304, -0.11772212984888597},
            {-12.207318389146511, -0.3859286365383454, -1.1388845342165084}
        };

        for (size_t i = 0; i < expected_log_responsibilities.size(); ++i) {
            for (size_t j = 0; j < k; ++j) {
                EXPECT_NEAR(log_responsibilities(i, j), expected_log_responsibilities[i][j], 1e-6)
                << "Mismatch at row " << i << ", col " << j;
            }
        }
    }
};

TEST_F(GMMTest_FriendAccess, ExpectationStep) {
    TestExpectationStep();
}

TEST_F(GMMTest_FriendAccess, ExpectationStep_SecondCase) {
    TestExpectationStep_SecondCase();
}
