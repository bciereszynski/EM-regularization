#include <set>
#include <algorithm>
#include "sampling.h"

struct VectorXdComparator {
    bool operator()(const Eigen::VectorXd &a, const Eigen::VectorXd &b) const {
        return std::lexicographical_compare(a.data(), a.data() + a.size(), b.data(), b.data() + b.size());
    }
};

std::pair<Eigen::MatrixXd, std::vector<int> > try_sampling_unique_data(
    std::mt19937 &rng,
    const Eigen::MatrixXd &data,
    const int k
) {
    const int n = static_cast<int>(data.rows());
    const int d = static_cast<int>(data.cols());

    std::set<Eigen::VectorXd, VectorXdComparator> unique_set;
    for (int i = 0; i < n; ++i) {
        unique_set.insert(data.row(i));
    }

    const int unique_size = static_cast<int>(unique_set.size());
    Eigen::MatrixXd unique_data(unique_size, d);
    int row = 0;
    for (const auto &vec: unique_set) {
        unique_data.row(row++) = vec;
    }

    std::vector<int> indices;

    if (unique_size < k) {
        std::vector<int> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0); // 0, 1, ..., n-1
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        indices.assign(all_indices.begin(), all_indices.begin() + k);

        return {data, indices};
    }

    std::vector<int> all_indices(unique_size);
    std::iota(all_indices.begin(), all_indices.end(), 0); // 0, 1, ..., unique_size-1
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    indices.assign(all_indices.begin(), all_indices.begin() + k);

    return {unique_data, indices};
}
