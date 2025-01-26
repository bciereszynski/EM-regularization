#include "GMM.h"
#include <iostream>
#include "sampling.h"

namespace {
    void initialize(GMMResult& result, const std::vector<std::vector<double>>& data, const std::vector<int>& indices, const bool verbose) {
        const int k = static_cast<int>(indices.size());
        const int d = static_cast<int>(data[0].size());

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < d; ++j) {
                result.clusters[i][j] = data[indices[i]][j];
            }
        }

        if (verbose) {
            std::cout << "Initial clusters:\n";
            for (int i = 0; i < k; ++i) {
                std::cout << "Cluster " << i + 1 << ": ";
                for (int j = 0; j < d; ++j) {
                    std::cout << result.clusters[i][j] << " ";
                }
                std::cout << "\n";
            }
        }
    }
}
//namespace

GMM::GMM(
    const double tolerance,
    const int max_iterations,
    const bool verbose,
    const unsigned int seed,
    const bool decompose_if_fails
) : tolerance(tolerance),
    max_iterations(max_iterations),
    verbose(verbose),
    rng(seed),
    decompose_if_fails(decompose_if_fails) {}

GMMResult GMM::fit(const std::vector<std::vector<double>>& data, GMMResult& result){
    if (verbose) {
        std::cout << "Starting GMM...\n";
    }
    return result;
}

GMMResult GMM::fit(const std::vector<std::vector<double>> &data, const int k) {
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());

    GMMResult result(d, n, k);

    if (n == 0) {
        return result;
    }
    assert(d > 0);
    assert(k > 0);
    assert(n >= k);

    //try sampling
    //initialize
    auto [sampled_data, indices] =  try_sampling_unique_data(rng, data, k);
    initialize(result, sampled_data, indices, verbose);
    fit(data, result);
    return result;
}
GMMResult GMM::fit(const std::vector<std::vector<double>> &data, const std::vector<int> &initial_clusters){
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());
    const int k = static_cast<int>(initial_clusters.size());

    GMMResult result(d, n, k);

    if (n == 0) {
        return result;
    }
    assert(d > 0);
    assert(k > 0);
    assert(n >= k);

    initialize(result, data, initial_clusters, verbose);
    GMM::fit(data, result);
    return result;
}
