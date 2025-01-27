#include <set>
#include <algorithm>
#include "sampling.h"

std::pair<std::vector<std::vector<double> >, std::vector<int> > try_sampling_unique_data(
    std::mt19937 &rng,
    const std::vector<std::vector<double> > &data,
    const int k
) {
    std::set<std::vector<double>, RowComparator> unique_set(data.begin(), data.end());
    std::vector<std::vector<double> > unique_data(unique_set.begin(), unique_set.end());
    int unique_size = static_cast<int>(unique_data.size());

    std::vector<int> indices;

    if (unique_size < k) {
        int n = static_cast<int>(data.size());
        std::vector<int> all_indices(n);
        std::iota(all_indices.begin(), all_indices.end(), 0); // 0, 1, .., n-1

        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        indices.assign(all_indices.begin(), all_indices.begin() + k);

        return {data, indices};
    }
    // else
    std::vector<int> all_indices(unique_size);
    std::iota(all_indices.begin(), all_indices.end(), 0); // 0, 1, .., unique_size-1

    std::shuffle(all_indices.begin(), all_indices.end(), rng);
    indices.assign(all_indices.begin(), all_indices.begin() + k);

    return {unique_data, indices};
}
