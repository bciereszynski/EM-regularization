#include <gtest/gtest.h>
#include "../src/algorithm/gmm/gmm.h"
#include "../src/data.h"
#include <Eigen/Dense>

TEST(GMMTest, BasicClustering) {
    Eigen::MatrixXd data(6, 1);
    data << 1.0, 1.1, 0.9, 5.0, 5.1, 4.9;

    GMM model(1e-4, 100, false, 42);
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

    GMM model(1e-6, 100, false, 42);

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
    GMM model(1e-6, 100, false, 42);
    Eigen::MatrixXd data(4, 2);
    data << 0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1;
    const auto result = model.fit(data, 2);
    const double sum = std::accumulate(result.weights.begin(), result.weights.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-8);
}

TEST(GMMTest, HandlesEmptyInput) {
    GMM model(1e-6, 100, false, 42);
    const Eigen::MatrixXd data;
    EXPECT_NO_THROW({
        const auto result = model.fit(data, 2);
        EXPECT_EQ(result.assignments.size(), 0);
        });
}

TEST(GMMTest, HandlesMinimalInputOnePoint) {
    GMM model(1e-6, 100, false, 42);
    Eigen::MatrixXd data(1, 2); // One point in 2D
    data << 1.0, 2.0;
    EXPECT_NO_THROW({
        const auto result = model.fit(data, 1);
        EXPECT_EQ(result.assignments.size(), 1);
        EXPECT_EQ(result.assignments[0], 0);
        });
}

TEST(GMMTest, TestFit) {
    Eigen::MatrixXd data(11, 2);
    data << 8.446952398696654, 11.512125077077995,
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

    constexpr int k = 3;
    constexpr int max_iterations = 1000;
    constexpr double tolerance = 0.001;

    const int n = data.rows();
    const int d = data.cols();
    GMMResult result(d, n, k);

    const GMM gmm(tolerance, max_iterations, true, 42);
    result.clusters = Eigen::MatrixXd(k, d);
    result.clusters << 14.398746744052609, 8.170480801024427,
            -17.86342608516553, -11.382757975845214,
            -23.45789229394996, -2.679611133069019;
    std::cout << result.clusters;
    result = gmm.fit(data, result);

    constexpr double expected_objective = -5.751353;
    EXPECT_NEAR(result.objective, expected_objective, 1e-6) << "Objective mismatch";
}

class GMMTest_FriendAccess : public ::testing::Test {
protected:
    GMM gmm{1e-3, 100, false, 42};

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
        result.clusters << 8.072821980338327, -18.486899691951532,
                -8.15593365254254, 15.568038996903327,
                3.2971078319133493, -15.064491056854397;
        result.weights = {0.3333333333333333, 0.3333333333333333, 0.3333333333333333};

        std::vector<Eigen::MatrixXd> precisions_cholesky(k, Eigen::MatrixXd::Identity(d, d));

        const auto [mean_log_likelihood, log_responsibilities] =
                GMM::expectation_step(data, k, result, precisions_cholesky);

        EXPECT_NEAR(mean_log_likelihood, -100.48916207565158, 1e-6);

        Eigen::MatrixXd expected_log_responsibilities(5, 3);
        expected_log_responsibilities <<
                -303.987598853661, 0.0, -220.36558061847856,
                -0.5640781381025022, -310.7288120901821, -0.8413788673385341,
                -66.73156857059266, -500.03062047627327, 0.0,
                -130.91721955111578, -179.62875633005083, 0.0,
                -93.5978919047501, 0.0, -49.83641435898852;

        for (int i = 0; i < 5; ++i) {
            EXPECT_TRUE(log_responsibilities.row(i).isApprox(expected_log_responsibilities.row(i), 1e-6))
            << "Mismatch in responsibility " << i << "\nExpected:\n" << expected_log_responsibilities.row(i)
            << "\nGot:\n" << log_responsibilities.row(i);
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

        const auto [mean_log_likelihood, log_responsibilities] =
                GMM::expectation_step(data, k, result, precisions_cholesky);

        EXPECT_NEAR(mean_log_likelihood, -7.5749208453554004, 1e-6);

        Eigen::MatrixXd expected_log_responsibilities(5, 3);
        expected_log_responsibilities <<
                -27.232032487796047, -0.04787476906218746, -3.063008541916525,
                -1.715062305720429, -4.3883879711018805, -0.21365526279087454,
                -14.510256021188855, -7.068564751881714, -0.0008523166008886918,
                -42.37473621066418, -2.1977119573849304, -0.11772212984888597,
                -12.207318389146511, -0.3859286365383454, -1.1388845342165084;

        for (int i = 0; i < expected_log_responsibilities.rows(); ++i) {
            EXPECT_TRUE(log_responsibilities.row(i).isApprox(expected_log_responsibilities.row(i), 1e-6))
                << "Mismatch at row " << i << "\nExpected:\n" << expected_log_responsibilities.row(i)
                << "\nGot:\n" << log_responsibilities.row(i);
        }
    }

    static void TestEstimateGaussian() {
        Eigen::MatrixXd data(11, 2);
        data << 8.446952398696654, 11.512125077077995,
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

        Eigen::MatrixXd responsibilities(11, 3);
        responsibilities << 1.0, 9.474243693098393e-255, 2.2081921561690226e-255,
                1.0, 5.7937342287019054e-229, 3.4206173646287936e-302,
                4.944828537633115e-259, 1.0, 9.289633056584679e-72,
                9.1422604084118e-310, 1.0, 5.7009547897076775e-24,
                1.0, 9.1422604084118e-310, 0.0,
                1.0, 5.130659100553779e-185, 2.4357668952132955e-276,
                1.0, 4.618920572015882e-102, 9.13459061470043e-140,
                0.9921502699429642, 1.1878799283334782e-70, 0.007849730057035107,
                1.3962696635361188e-53, 1.0, 2.4031087966987318e-86,
                1.0, 3.669e-320, 0.0,
                0.0, 5.7009547897076775e-24, 1.0;

        const std::vector<double> expected_weights = {0.6356500245402693, 0.2727272727272728, 0.09162270273245789};

        Eigen::MatrixXd expected_centers(3, 2);
        expected_centers << 9.521802600275707, 4.510367687529266,
                -8.371544247345165, -14.996543900509586,
                -23.344566457264932, -2.514909854874149;

        std::vector<Eigen::MatrixXd> expected_covariances = {
            (Eigen::MatrixXd(2, 2) << 72.47093182603915, -47.76866097711419,
             -47.76866097711419, 71.38576144184982).finished(),
            (Eigen::MatrixXd(2, 2) << 69.99901630029576, -8.982124169630735,
             -8.982124169630735, 9.204054778365268).finished(),
            (Eigen::MatrixXd(2, 2) << 1.6360747652529521, 2.3777773272343468,
             2.3777773272343468, 3.4557253360212914).finished()
        };

        EmpiricalRegularizer regularizer;

        auto [computed_weights, computed_centers, computed_covariances] =
                GMM::estimate_gaussian_parameters(data, 3, responsibilities, regularizer);

        ASSERT_EQ(computed_weights.size(), expected_weights.size());
        for (size_t i = 0; i < computed_weights.size(); ++i) {
            ASSERT_NEAR(computed_weights[i], expected_weights[i], 1e-6) << "Weight mismatch at index " << i;
        }

        ASSERT_TRUE(computed_centers.isApprox(expected_centers, 1e-6)) << "Centers do not match.";

        ASSERT_EQ(computed_covariances.size(), expected_covariances.size());
        for (size_t i = 0; i < computed_covariances.size(); ++i) {
            ASSERT_TRUE(computed_covariances[i].isApprox(expected_covariances[i], 1e-6))
            << "Covariance matrix mismatch at index " << i;
        }
    }

    static void Test_Maximisation_Step() {
        Eigen::MatrixXd data(11, 2);
        data << 8.446952398696654, 11.512125077077995,
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

        Eigen::MatrixXd log_responsibilities(11, 3);
        log_responsibilities << 0.0, -584.9106217870079, -586.367024561603,
                0.0, -525.5352092674308, -694.150877033445,
                -594.7711967948704, 0.0, -163.55722764227863,
                -711.5884711638098, 0.0, -53.52140856407806,
                0.0, -711.5884711638098, -775.4248071159499,
                0.0, -424.34300807332846, -634.6232240127079,
                0.0, -233.33351845001846, -320.1498446453707,
                -0.007880701372130261, -161.00878636417707, -4.847276135417928,
                -121.70320577427677, 0.0, -197.1455547663949,
                0.0, -735.5273015257319, -837.7696318031681,
                -775.4248071159499, -53.52140856407806, 0.0;

        const std::vector<double> expected_weights = {0.6356500245402693, 0.2727272727272728, 0.09162270273245789};

        Eigen::MatrixXd expected_centers(3, 2);
        expected_centers << 9.521802600275707, 4.510367687529266,
                -8.371544247345165, -14.996543900509586,
                -23.344566457264932, -2.514909854874149;

        const std::vector<Eigen::MatrixXd> expected_covariances = {
            (Eigen::MatrixXd(2, 2) << 72.47093182603915, -47.76866097711419,
             -47.76866097711419, 71.38576144184982).finished(),
            (Eigen::MatrixXd(2, 2) << 69.99901630029576, -8.982124169630735,
             -8.982124169630735, 9.204054778365268).finished(),
            (Eigen::MatrixXd(2, 2) << 1.6360747652529521, 2.3777773272343468,
             2.3777773272343468, 3.4557253360212914).finished()
        };

        GMMResult result(2, 11, 3);
        std::vector<Eigen::MatrixXd> precision_cholesky(3, Eigen::MatrixXd::Identity(2, 2));

        const GMM gmm;
        gmm.maximization_step(data, 3, result, log_responsibilities, precision_cholesky);

        for (int i = 0; i < 3; ++i) {
            EXPECT_NEAR(result.weights[i], expected_weights[i], 1e-6) << "Mismatch in weight " << i;
        }

        EXPECT_TRUE(result.clusters.isApprox(expected_centers, 1e-6))
        << "Mismatch in centers\nExpected:\n" << expected_centers << "\nGot:\n" << result.clusters;

        for (int i = 0; i < 3; ++i) {
            EXPECT_TRUE(result.covariances[i].isApprox(expected_covariances[i], 1e-6))
            << "Mismatch in covariance " << i << "\nExpected:\n" << expected_covariances[i]
            << "\nGot:\n" << result.covariances[i];
        }
    }
};

TEST_F(GMMTest_FriendAccess, ExpectationStep) {
    TestExpectationStep();
}

TEST_F(GMMTest_FriendAccess, ExpectationStep_SecondCase) {
    TestExpectationStep_SecondCase();
}

TEST_F(GMMTest_FriendAccess, TestEstimateGaussian) {
    TestEstimateGaussian();
}

TEST_F(GMMTest_FriendAccess, TestMaximisationStep) {
    Test_Maximisation_Step();
}
