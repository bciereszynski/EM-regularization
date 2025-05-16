#include <gtest/gtest.h>
#include "../src/algorithm/GeneticalAlgorithm.h"

class GATest_FriendAccess : public ::testing::Test {
protected:
    GeneticalAlgorithm ga{std::mt19937(42), 100, 100, 10, 100, true};

    static void TestDistance() {
        GMMResult a(2, 1, 2);
        GMMResult b(2, 1, 2);

        a.covariances[0] << 24.475167513040603, 3.89359255061906,
                3.89359255061906, 56.46934534189046;

        b.covariances[0] << 75.72373252043417, 26.909455437271575,
                26.909455437271575, 114.94144383397848;

        a.clusters.row(0) << -13.268858005520594, -8.200086911892441;
        b.clusters.row(0) << 2.495204896010447, 3.12530361824172;

        constexpr double expected_distance1 = 28942.35148764443;

        const double dist = GeneticalAlgorithm::distance(a, 0, b, 0);

        EXPECT_NEAR(dist, expected_distance1, 1e-6) << "Mismatch in distance";
    }
};

TEST_F(GATest_FriendAccess, TestDistance) {
    TestDistance();
}
