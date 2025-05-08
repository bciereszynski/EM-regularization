#include <gtest/gtest.h>
#include "../src/algorithm/gmm/gmm.h"
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
