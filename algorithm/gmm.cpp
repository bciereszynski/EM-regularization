#include "GMM.h"
#include <iostream>

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
    fit(data, result);
    return result;
}
GMMResult GMM::fit(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &initial_clusters){
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());
    const int k = static_cast<int>(initial_clusters[0].size());

    GMMResult result(d, n, k);

    if (n == 0) {
        return result;
    }
    assert(d > 0);
    assert(k > 0);
    assert(n >= k);

    //initialize
    GMM::fit(data, result);
    return result;
}
