#include "GMM.h"
#include <chrono>
#include <iostream>
#include <limits>

#include "regularization/EmpiricalRegularizer.h"
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
    auto precision_cholesky = std::vector<Eigen::MatrixXd>(k, Eigen::MatrixXd(d, d));

    computePrecisionCholesky(result, precision_cholesky);

    if (n == k) {
        result.converged = true;
        result.iterations = 1;
    } else
        for (int i = 0; i < max_iterations; ++i) {
            double previous_objective = result.objective;

            const auto t0 = std::chrono::system_clock::now();
            auto [objective, log_responsibilities] = expectation(data, k, result, precision_cholesky);
            maximization_step(data, k, result, log_responsibilities, precision_cholesky);
            const auto t1 = std::chrono::system_clock::now();

            result.objective = objective;
            const auto change = result.objective - previous_objective;

            if (verbose) {
                std::cout << "Iteration " << i + 1 << ": ";
                std::cout << "Change:" << change << "\t";
                std::cout << "Objective:" << result.objective << "\t";
                std::cout << "Time: " << (t1 - t0).count() << "\t";
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
                 const std::vector<Eigen::MatrixXd> &precisionsCholesky) {
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

void GMM::maximization_step(const std::vector<std::vector<double> > &data, int k, GMMResult &result,
                            const std::vector<std::vector<double> > &log_responsibilities,
                            std::vector<Eigen::MatrixXd> &precision_cholesky) {
    std::vector<std::vector<double> > responsibilities(log_responsibilities.size(),
                                                       std::vector<double>(log_responsibilities[0].size()));
    for (size_t i = 0; i < log_responsibilities.size(); ++i) {
        for (size_t j = 0; j < log_responsibilities[i].size(); ++j) {
            responsibilities[i][j] = std::exp(log_responsibilities[i][j]);
        }
    }
    std::tie(result.weights, result.clusters, result.covariances) = estimate_gaussian_parameters(
        data, k, responsibilities);
    computePrecisionCholesky(result, precision_cholesky);
}

std::tuple<std::vector<double>, std::vector<std::vector<double> >, std::vector<Eigen::Matrix<double, Eigen::Dynamic,
    Eigen::Dynamic> > >
GMM::estimate_gaussian_parameters(
    const std::vector<std::vector<double> > &data,
    int k,
    std::vector<std::vector<double> > &responsibilities
) {
    const int n = data.size();
    const int d = data[0].size();
    const double eps = 10 * std::numeric_limits<double>::epsilon();

    std::vector<double> weights(k, eps);

    for (int i = 0; i < k; ++i) {
        double sum_resp = 0.0;
        for (int j = 0; j < n; ++j) {
            sum_resp += responsibilities[j][i];
        }

        if (sum_resp < 1e-32) {
            for (int j = 0; j < n; ++j) {
                responsibilities[j][i] = 1.0 / n;
            }
        }

        for (int j = 0; j < n; ++j) {
            weights[i] += responsibilities[j][i];
        }
    }

    const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double &w: weights) {
        w /= total_weight;
    }

    std::vector<std::vector<double> > clusters(k, std::vector<double>(d, 0.0));
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > covariances(k);

    for (int i = 0; i < k; ++i) {
        std::vector<double> responsibility_column(n, 0.0);
        for (int j = 0; j < n; ++j) {
            responsibility_column[j] = responsibilities[j][i];
        }

        EmpiricalRegularizer regularizer = EmpiricalRegularizer();
        std::vector<std::vector<double> > covariances_temp;
        std::vector<double> cluster_temp;
        std::tie(covariances_temp, cluster_temp) = regularizer.fit(data, responsibility_column);

        clusters[i] = cluster_temp;

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cov_matrix(d, d);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                cov_matrix(r, c) = covariances_temp[r][c];
            }
        }
        covariances[i] = cov_matrix;
    }

    return std::make_tuple(weights, clusters, covariances);
}

std::vector<std::vector<double> > GMM::estimateWeightedLogProbabilities(
    const std::vector<std::vector<double> > &data, const int k, const GMMResult &result,
    const std::vector<Eigen::MatrixXd> &precisionsCholesky
) {
    const int n = static_cast<int>(data.size());
    const int d = static_cast<int>(data[0].size());

    std::vector<double> log_det_cholesky(k, 0.0);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            log_det_cholesky[i] += log(precisionsCholesky[i](j, j));
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
                    y[row] += data[j][col] * precisionsCholesky[i](col, row) -
                            result.clusters[i][col] * precisionsCholesky[i](col, row);
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

void GMM::computePrecisionCholesky(GMMResult &result,
                                   std::vector<Eigen::MatrixXd> &precisions_cholesky) {
    const int k = result.covariances.size();
    const int d = result.covariances[0].rows();
    for (int i = 0; i < k; ++i) {
        try {
            Eigen::LLT<Eigen::MatrixXd> cholesky(result.covariances[i]);
            precisions_cholesky[i] = cholesky.matrixU().transpose().solve(Eigen::MatrixXd::Identity(d, d));
        } catch (const std::exception &e) {
            if (decompose_if_fails) {
                Eigen::EigenSolver<Eigen::MatrixXd> eig(result.covariances[i]);
                Eigen::MatrixXd eigenvalues = eig.eigenvalues().real().cwiseMax(1e-6).asDiagonal();
                Eigen::MatrixXd eigenvectors = eig.eigenvectors().real();

                result.covariances[i] = eigenvectors * eigenvalues * eigenvectors.transpose();

                Eigen::LLT<Eigen::MatrixXd> cholesky_new(result.covariances[i]);
                precisions_cholesky[i] = cholesky_new.matrixU().transpose().solve(Eigen::MatrixXd::Identity(d, d));
            } else {
                throw std::runtime_error("GMM Failed: " + std::string(e.what()));
            }
        }
    }
}
