#include <gtest/gtest.h>
#include "../algorithm/gmm/gmm.h"
#include <Eigen/Dense>

TEST(GMMTest, BasicClustering) {
    Eigen::MatrixXd data(6, 1);
    data << 1.0, 1.1, 0.9, 5.0, 5.1, 4.9;

    GMM model(1e-4, 100, false, 1234, true);
    auto result = model.fit(data, 2);

    EXPECT_EQ(result.assignments.size(), 6);
    EXPECT_EQ(result.weights.size(), 2);
    EXPECT_EQ(result.clusters.size(), 2);
    EXPECT_EQ(result.covariances.size(), 2);

    for (const auto &cov: result.covariances) {
        EXPECT_GT(cov(0, 0), 0);
    }
}

