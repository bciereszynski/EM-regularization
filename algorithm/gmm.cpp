#include "GMM.h"
#include <chrono>
#include <iostream>
#include <limits>

#include "sampling.h"

namespace {
    void initialize(GMMResult &result, const std::vector<std::vector<double> > &data, const std::vector<int> &indices,
                    const bool verbose) {
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
    decompose_if_fails(decompose_if_fails) {
}

GMMResult GMM::fit(const std::vector<std::vector<double> > &data, GMMResult &result) {
    if (verbose) {
        std::cout << "Starting GMM...\n";
    }
    const auto start_time = std::chrono::system_clock::now();
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());
    const int k = static_cast<int>(result.clusters.size());

    result.iterations = max_iterations;
    result.converged = false;
    double result_objective = -std::numeric_limits<double>::infinity();
    double previous_objective;

    auto log_responsibilities = std::vector<std::vector<double> >(n, std::vector<double>(k, 0.0));
    auto precision_cholesky = std::vector<std::vector<std::vector<double> > >(k,
                                                                              std::vector<std::vector<double> >(
                                                                                  d, std::vector<double>(d, 0.0)));
    // compute cholesky

    if (n == k) {
        result.converged = true;
        result.iterations = 1;
    } else
        for (int i = 0; i < max_iterations; ++i) {
            previous_objective = result_objective;

            const auto t0 = std::chrono::system_clock::now();
            // expectation
            // maximisation
            const auto t1 = std::chrono::system_clock::now();

            const auto change = result_objective - previous_objective;

            if (verbose) {
                std::cout << "Iteration " << i + 1 << ": ";
                std::cout << "Change:" << change << "\t";
                std::cout << "Objective:" << result.objective << "\t";
                std::cout << "Time: " << t1 - t0 << "\t";
                std::cout << std::endl;
            }

            if (change < tolerance) {
                result.converged = true;
                result.iterations = i;
                break;
            }
        }

    // assign points to clusters

    result.elapsed = std::chrono::duration_cast<std::chrono::duration<double> >(
        std::chrono::system_clock::now() - start_time).count();
    return result;
}

GMMResult GMM::fit(const std::vector<std::vector<double> > &data, const int k) {
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());

    GMMResult result(d, n, k);

    if (n == 0) {
        return result;
    }
    assert(d > 0);
    assert(k > 0);
    assert(n >= k);

    auto [sampled_data, indices] = try_sampling_unique_data(rng, data, k);
    initialize(result, sampled_data, indices, verbose);
    fit(data, result);
    return result;
}

GMMResult GMM::fit(const std::vector<std::vector<double> > &data, const std::vector<int> &initial_clusters) {
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
    fit(data, result);
    return result;
}
