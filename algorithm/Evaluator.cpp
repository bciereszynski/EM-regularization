//
// Created by bcier on 3/19/2025.
//

#include "Evaluator.h"
#include <Eigen/Dense>
#include <algorithm>

#include <utility>

namespace {
    std::pair<std::vector<int>, int> find_nearest_centroids(
        const Eigen::MatrixXd &matrix_a,
        const Eigen::MatrixXd &matrix_b) {
        int k = matrix_a.rows();

        std::vector<int> nearest_centroids(k);

        for (int i = 0; i < k; ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int nearest_index = -1;

            for (int j = 0; j < k; ++j) {
                double distance = (matrix_a.row(i) - matrix_b.row(j)).norm();

                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_index = j;
                }
            }

            nearest_centroids[i] = nearest_index;
        }

        return {nearest_centroids, k};
    }

    int orphans_count(const Eigen::MatrixXd &matrix_a, const Eigen::MatrixXd &matrix_b) {
        auto [nearest_centroids, k] = find_nearest_centroids(matrix_a, matrix_b);

        std::vector<int> centroid_assignments(k, 0);

        for (int i = 0; i < k; ++i) {
            centroid_assignments[nearest_centroids[i]]++;
        }

        int orphan_count = 0;
        for (int i = 0; i < k; ++i) {
            if (centroid_assignments[i] == 0) {
                orphan_count++;
            }
        }

        return orphan_count;
    }

    int centroid_index(const Eigen::MatrixXd &matrix_a, const Eigen::MatrixXd &matrix_b) {
        const int centroid_index1 = orphans_count(matrix_a, matrix_b);
        const int centroid_index2 = orphans_count(matrix_b, matrix_a);

        return std::max(centroid_index1, centroid_index2);
    }
}

Evaluator::Evaluator(const std::vector<std::vector<double> > &data, const std::vector<int> &expected_clusters,
                     const int k,
                     GMMResult result): data(data), expected_clusters(expected_clusters), k(k),
                                        result(std::move(result)) {
}

int Evaluator::centroid_index_sphere() {
    Eigen::MatrixXd matrix_a(k, data[0].size());
    Eigen::MatrixXd matrix_b(k, data[0].size());

    for (int i = 0; i < k; ++i) {
        std::vector<Eigen::VectorXd> points;
        for (size_t j = 0; j < data.size(); ++j) {
            if (expected_clusters[j] == i) {
                points.emplace_back(Eigen::VectorXd::Map(data[j].data(), data[j].size()));
            }
        }
        if (!points.empty()) {
            Eigen::MatrixXd points_matrix(points.size(), data[0].size());
            for (size_t j = 0; j < points.size(); ++j) {
                points_matrix.row(j) = points[j];
            }
            matrix_a.row(i) = points_matrix.colwise().mean();
        }
    }

    for (int i = 0; i < k; ++i) {
        matrix_b.row(i) = Eigen::VectorXd::Map(result.clusters[i].data(), result.clusters[i].size());
    }

    return centroid_index(matrix_a, matrix_b);
}

