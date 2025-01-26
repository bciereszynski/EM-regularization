#include "GMM.h"
#include <chrono>
#include <iostream>
#include <limits>

#include "mathematical.h"
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
    result.objective = -std::numeric_limits<double>::infinity();

    auto log_responsibilities = std::vector<std::vector<double> >(n, std::vector<double>(k, 0.0));
    auto precision_cholesky =
            std::vector<std::vector<std::vector<double> > >(
                k, std::vector<std::vector<double> >(d, std::vector<double>(d, 0.0)));
    compute_precision_cholesky(result, precisions_cholesky);

    if (n == k) {
        result.converged = true;
        result.iterations = 1;
    } else
        for (int i = 0; i < max_iterations; ++i) {
            double previous_objective = result.objective;

            const auto t0 = std::chrono::system_clock::now();
            auto [objective, log_responsibilities] = expectation(data, k, result, precision_cholesky);
            // maximization_step(data, k, result, log_responsibilities, precisions_cholesky);
            const auto t1 = std::chrono::system_clock::now();

            result.objective = objective;
            const auto change = result.objective - previous_objective;

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

std::pair<double, std::vector<std::vector<double> > >
GMM::expectation(const std::vector<std::vector<double> > &data,
                 const int k, const GMMResult &result,
                 const std::vector<std::vector<std::vector<double> > > &precisionsCholesky) {
    const int n = static_cast<int>(data.size());

    const auto weighted_log_probabilities = estimateWeightedLogProbabilities(data, k, result, precisionsCholesky);

    auto log_probabilities_norm = std::vector<double>(n, 0.0);

    for (int i = 0; i < n; ++i) {
        log_probabilities_norm[i] = log_sum_exp(weighted_log_probabilities[i]);
    }

    std::vector<std::vector<double> > log_responsibilities(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            log_responsibilities[i][j] = weighted_log_probabilities[i][j] - log_probabilities_norm[i];
        }
    }
    double mean = std::accumulate(log_probabilities_norm.begin(), log_probabilities_norm.end(), 0.0) /
                  log_probabilities_norm.size();

    return {mean, log_responsibilities};
}

std::vector<std::vector<double> > GMM::estimateWeightedLogProbabilities(
    const std::vector<std::vector<double> > &data, const int k, const GMMResult &result,
    const std::vector<std::vector<std::vector<double> > > &precisionsCholesky
) {
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());

    std::vector<double> log_det_cholesky(k, 0.0);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            log_det_cholesky[i] += log(precisionsCholesky[i][j][j]);
        }
    }

    std::vector<std::vector<double> > log_probabilities(n, std::vector<double>(k, 0.0));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            std::vector<double> y(d);
            // y = data[j] * precisionsCholesky[i] - (result.clusters[i]' * precisionsCholesky[i])
            for (int row = 0; row < d; ++row) {
                y[row] = 0.0;
                for (int col = 0; col < d; ++col) {
                    y[row] += data[j][col] * precisionsCholesky[i][col][row] -
                            result.clusters[i][col] * precisionsCholesky[i][col][row];
                }
            }
            double sum = 0.0;
            for (const double val: y) {
                sum += val * val;
            }
            log_probabilities[j][i] = sum;
        }
    }
    std::vector<std::vector<double> > result_probabilities(n, std::vector<double>(k));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            result_probabilities[j][i] = -0.5 * (d * std::log(2 * M_PI) + log_probabilities[j][i]) +
                                         log_det_cholesky[i] + std::log(result.weights[i]);
        }
    }

    return result_probabilities;
}
