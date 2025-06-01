#include "GMM.h"
#include <chrono>
#include <iostream>
#include <limits>

#include "./helpers/mathematical.h"

namespace {
    int assign_to_cluster(const int point, const Eigen::MatrixXd &probabilities,
                          std::vector<bool> is_empty) {
        const int k = probabilities.cols();

        int max_cluster = 0;
        double max_probability = -std::numeric_limits<double>::infinity();

        for (int cluster = 0; cluster < k; ++cluster) {
            const double probability = probabilities(point, cluster);

            if (probability > max_probability) {
                max_cluster = cluster;
                max_probability = probability;
            } else if (probability == max_probability) {
                if (is_empty[cluster] && !is_empty[max_cluster]) {
                    max_cluster = cluster;
                    max_probability = probability;
                }
            }
        }

        return max_cluster;
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

GMM::GMM(
    const std::mt19937 &rng,
    const double tolerance,
    const int max_iterations,
    const bool verbose,
    const bool decompose_if_fails
) : tolerance(tolerance),
    max_iterations(max_iterations),
    verbose(verbose),
    rng(rng),
    decompose_if_fails(decompose_if_fails) {
}

// TODO flaga eigen - blas
GMMResult GMM::fit(const std::vector<std::vector<double> > &data, int k) {
    Eigen::MatrixXd data_matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        data_matrix.row(i) = Eigen::VectorXd::Map(data[i].data(), data[i].size());
    }
    return fit(data_matrix, k);
}

GMMResult GMM::fit(const Eigen::MatrixXd &data, GMMResult &result) const {
    if (verbose) {
        std::cout << "Starting GMM...\n";
    }
    const auto start_time = std::chrono::system_clock::now();
    const int n = data.rows();
    const int d = data.cols();
    const int k = static_cast<int>(result.clusters.rows());

    result.iterations = max_iterations;
    result.converged = false;
    result.objective = -std::numeric_limits<double>::infinity();

    Eigen::MatrixXd log_responsibilities(n, k);
    auto precision_cholesky = std::vector<Eigen::MatrixXd>(k, Eigen::MatrixXd(d, d));

    compute_precisions_cholesky(result, precision_cholesky);

    if (n == k) {
        result.converged = true;
        result.iterations = 1;
    } else
        for (int i = 0; i < max_iterations; ++i) {
            const double previous_objective = result.objective;

            const auto t0 = std::chrono::system_clock::now();
            auto [objective, temp_log_responsibilities] = expectation_step(data, k, result, precision_cholesky);
            log_responsibilities = temp_log_responsibilities;
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

    const auto probabilities = estimate_weighted_log_probabilities(data, k, result, precision_cholesky);

    std::vector<bool> is_empty(k, true);
    for (int i = 0; i < n; ++i) {
        const int cluster = assign_to_cluster(i, probabilities, is_empty);
        is_empty[cluster] = false;
        result.assignments[i] = cluster;
    }
    // TODO hight resolution clock
    result.elapsed = std::chrono::duration_cast<std::chrono::duration<double> >(
        std::chrono::system_clock::now() - start_time).count();
    return result;
}

GMMResult GMM::fit(const Eigen::MatrixXd &data, const int k) {
    const int n = data.rows();
    const int d = data.cols();

    GMMResult result(d, n, k);

    if (n == 0) {
        return result;
    }
    assert(d > 0);
    assert(k > 0);
    assert(n >= k);

    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);

    std::shuffle(permutation.begin(), permutation.end(), rng);

    result.clusters = Eigen::MatrixXd(k, d);
    for (int i = 0; i < k; ++i) {
        result.clusters.row(i) = data.row(permutation[i]);
    }
    fit(data, result);
    return result;
}

std::pair<double, Eigen::MatrixXd>
GMM::expectation_step(const Eigen::MatrixXd &data,
                      const int k, const GMMResult &result,
                      const std::vector<Eigen::MatrixXd> &precisionsCholesky) {
    const int n = data.rows();

    const auto weighted_log_probabilities =
            estimate_weighted_log_probabilities(data, k, result, precisionsCholesky);

    auto log_probabilities_norm = std::vector<double>(n, 0.0);

    for (int i = 0; i < n; ++i) {
        log_probabilities_norm[i] = log_sum_exp(weighted_log_probabilities.row(i));
    }

    Eigen::MatrixXd log_responsibilities(n, k);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            log_responsibilities(i, j) = weighted_log_probabilities(i, j) - log_probabilities_norm[i];
        }
    }
    double mean = std::accumulate(log_probabilities_norm.begin(), log_probabilities_norm.end(), 0.0) /
                  log_probabilities_norm.size();

    // std::cout << "weighted_log_probabilities:\n";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << "Row " << i << ": ";
    //     for (int j = 0; j < k; ++j) {
    //         std::cout << weighted_log_probabilities[i][j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    //
    // std::cout << "log_probabilities_norm:\n";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << "i=" << i << ": " << log_probabilities_norm[i] << "\n";
    // }
    //
    // std::cout << "log_responsibilities:\n" << log_responsibilities << "\n";
    //
    // std::cout << "mean: " << mean << "\n";

    return {mean, log_responsibilities};
}

void GMM::maximization_step(const Eigen::MatrixXd &data, const int k, GMMResult &result,
                            const Eigen::MatrixXd &log_responsibilities,
                            std::vector<Eigen::MatrixXd> &precision_cholesky) const {
    Eigen::MatrixXd responsibilities = log_responsibilities.array().exp();
    auto regularizer = EmpiricalRegularizer();
    std::tie(result.weights, result.clusters, result.covariances) = GMM::estimate_gaussian_parameters(
        data, k, responsibilities, regularizer);
    compute_precisions_cholesky(result, precision_cholesky);
}

void GMM::compute_precisions_cholesky(GMMResult &result,
                                      std::vector<Eigen::MatrixXd> &precisions_cholesky) const {
    const int k = result.covariances.size();
    const int d = result.covariances[0].rows();
    for (int i = 0; i < k; ++i) {
        try {
            Eigen::LLT<Eigen::MatrixXd> cholesky(result.covariances[i]);
            precisions_cholesky[i] = cholesky.matrixU().solve(Eigen::MatrixXd::Identity(d, d));
        } catch (const std::exception &e) {
            if (decompose_if_fails) {
                Eigen::EigenSolver<Eigen::MatrixXd> eig(result.covariances[i]);
                Eigen::MatrixXd eigenvalues = eig.eigenvalues().real().cwiseMax(1e-6).asDiagonal();
                Eigen::MatrixXd eigenvectors = eig.eigenvectors().real();

                result.covariances[i] = eigenvectors * eigenvalues * eigenvectors.transpose();

                Eigen::LLT<Eigen::MatrixXd> cholesky_new(result.covariances[i]);
                precisions_cholesky[i] = cholesky_new.matrixU().solve(Eigen::MatrixXd::Identity(d, d));
            } else {
                throw std::runtime_error("GMM Failed: " + std::string(e.what()));
            }
        }
    }
}

std::tuple<std::vector<double>, Eigen::MatrixXd,
    std::vector<Eigen::MatrixXd> >
GMM::estimate_gaussian_parameters(const Eigen::MatrixXd &data, const int k,
                                  Eigen::MatrixXd &responsibilities,
                                  CovarianceMatrixRegularizer &regularizer) {
    const int n = data.rows();
    const int d = data.cols();
    constexpr double eps = 10 * std::numeric_limits<double>::epsilon();

    std::vector<double> weights(k, eps);

    for (int i = 0; i < k; ++i) {
        double sum_resp = responsibilities.col(i).sum();

        if (sum_resp < 1e-32) {
            responsibilities.col(i).setConstant(1.0 / n);
        }

        weights[i] = responsibilities.col(i).sum();
    }

    const double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double &w: weights) {
        w /= total_weight;
    }

    Eigen::MatrixXd clusters(k, d);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > covariances(k);

    for (int i = 0; i < k; ++i) {
        std::vector<double> responsibility_column(responsibilities.col(i).data(),
                                                  responsibilities.col(i).data() + n);

        Eigen::MatrixXd covariances_temp;
        Eigen::VectorXd cluster_temp;
        std::tie(covariances_temp, cluster_temp) = regularizer.fit(data, responsibility_column);

        clusters.row(i) = cluster_temp.transpose();
        covariances[i] = covariances_temp;
    }

    return std::make_tuple(weights, clusters, covariances);
}
