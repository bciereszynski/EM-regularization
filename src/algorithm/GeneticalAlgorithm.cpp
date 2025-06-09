//
// Created by bcier on 3/22/2025.
//

#include "GeneticalAlgorithm.h"
#include "SqMahalanobis.h"


#include <algorithm>
#include "../external/hungarian/hungarian.h"
#include "gmm/gmm.h"
#include <chrono>


double GeneticalAlgorithm::distance(const GMMResult &a, const int i, const GMMResult &b, const int j) {
    const SqMahalanobis sqma(a.covariances[i], true);
    const SqMahalanobis sqmb(b.covariances[j], true);

    const double distance1 = Distances::evaluate(sqma, a.clusters.row(i), b.clusters.row(j));
    const double distance2 = Distances::evaluate(sqmb, a.clusters.row(i), b.clusters.row(j));

    return (distance1 + distance2) / 2.0;
}

GMMResult GeneticalAlgorithm::run(const std::vector<std::vector<double> > &data, int k) {
    Eigen::MatrixXd data_matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        data_matrix.row(i) = Eigen::VectorXd::Map(data[i].data(), data[i].size());
    }
    return run(data_matrix, k);
}

GMMResult GeneticalAlgorithm::run(const Eigen::MatrixXd &data, const int k) {
    auto best_obj = -std::numeric_limits<double>::infinity();
    pop.clear();
    auto iterations_without_improvement = 0;

    auto gmm = GMM(rng);

    for (int i = 0; i < pop_max_size; ++i) {
        GMMResult result = gmm.fit(data, k);
        pop.add(result);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int iter;
    for (iter = 0; iter < max_iterations; ++iter) {
        auto [parent1, parent2] = binary_tournament();

        auto child = crossover(parent1, parent2, data);

        mutate(child, data);

        // local search
        gmm.fit(data, child);
        pop.add(child);

        // reduce population size
        auto size = pop.active_size();
        if (size > pop_max_size) {
            pop.eliminate(size - pop_min_size, rng);
        }

        // check for improvement
        const auto best_solution = get_best_solution();
        if (best_solution.objective > best_obj) {
            best_obj = best_solution.objective;
            iterations_without_improvement = 0;
        } else {
            ++iterations_without_improvement;
        }

        if (iterations_without_improvement > max_iterations_without_improvement) {
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    GMMResult best_solution = get_best_solution();
    best_solution.elapsed = elapsed_time.count();
    best_solution.iterations = iter;

    return best_solution;
}

std::pair<GMMResult, GMMResult> GeneticalAlgorithm::binary_tournament() {
    std::vector<int> valid_indices(pop.size());
    std::iota(valid_indices.begin(), valid_indices.end(), 0);

    valid_indices.erase(
        std::remove_if(valid_indices.begin(), valid_indices.end(),
                       [&](const int i) { return !pop.is_active(i); }),
        valid_indices.end()
    );

    if (valid_indices.size() < 4) {
        throw std::runtime_error("Not enough active individuals for tournament selection.");
    }
    std::shuffle(valid_indices.begin(), valid_indices.end(), rng);

    const GMMResult &parent1 = pop[valid_indices[0]];
    const GMMResult &parent2 = pop[valid_indices[1]];
    const GMMResult &parent3 = pop[valid_indices[2]];
    const GMMResult &parent4 = pop[valid_indices[3]];

    return {
        is_better(parent1, parent2) ? parent1 : parent2,
        is_better(parent3, parent4) ? parent3 : parent4
    };
}

GMMResult GeneticalAlgorithm::get_best_solution() {
    if (pop.active_size() == 0) {
        throw std::runtime_error("No active solutions in population.");
    }

    GMMResult best_solution = pop[0];
    bool found = false;

    for (size_t i = 0; i < pop.size(); ++i) {
        if (pop.is_active(i) && (!found || is_better(pop[i], best_solution))) {
            best_solution = pop[i];
            found = true;
        }
    }

    return best_solution;
}

void GeneticalAlgorithm::Population::eliminate(const size_t to_remove, std::mt19937 &rng) {
    size_t removed = 0;
    size_t n = size();

    for (size_t i = 0; i < n; ++i) {
        if (to_remove == removed) {
            break;
        }

        for (size_t j = i + 1; j < n; ++j) {
            if (to_remove == removed) {
                break;
            }

            if (is_active(i) && is_active(j)) {
                if (std::abs(population[i].objective - population[j].objective) < 1e-9) {
                    removed++;
                    if (std::uniform_real_distribution<>(0.0, 1.0)(rng) > 0.5) {
                        remove(i);
                    } else {
                        remove(j);
                    }
                }
            }
        }
    }

    if (to_remove > removed) {
        n = size();
        const size_t remaining = active_size() - (to_remove - removed);

        if (remaining >= active_size()) {
            return;
        }

        std::vector<GMMResult> active_population;
        for (size_t i = 0; i < n; ++i) {
            if (is_active(i)) {
                active_population.push_back(population[i]);
            }
        }

        std::sort(active_population.begin(), active_population.end(), is_better); // TODO - replace sort?

        const GMMResult threshold = active_population[remaining];

        for (size_t i = 0; i < n; ++i) {
            if (is_active(i) && is_better(threshold, population[i])) {
                remove(i);
            }
        }
    }
}

void GeneticalAlgorithm::mutate(GMMResult &individual, const Eigen::MatrixXd &data) {
    const int k = individual.k;
    const int n = data.rows();
    const int d = data.cols();

    if (n > 0 && k > 0) {
        std::uniform_int_distribution<int> dist_k(0, k - 1);
        std::uniform_int_distribution<int> dist_n(0, n - 1);

        const int to = dist_k(rng);
        const int from = dist_n(rng);

        individual.clusters.row(to) = data.row(from);

        double sum_det = 0.0;
        for (int j = 0; j < k; ++j) {
            sum_det += individual.covariances[j].determinant();
        }
        const double m = sum_det / k;
        const double value = (m > 0 ? m : 1.0) * (1.0 / d);

        individual.covariances[to] = Eigen::MatrixXd::Identity(d, d) * value;

        individual.reset();
    }
}

GMMResult GeneticalAlgorithm::crossover(const GMMResult &parent1, const GMMResult &parent2,
                                        const Eigen::MatrixXd &data) {
    const int k = parent1.k;

    std::vector<std::vector<double> > weights(k, std::vector<double>(k));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            weights[i][j] = distance(parent1, i, parent2, j);
        }
    }

    std::vector<int> assignment;

    auto h_alg = HungarianAlgorithm();
    h_alg.Solve(weights, assignment);

    GMMResult offspring = parent1;
    offspring.reset();

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < k; ++i) {
        if (dist(rng) > 0.5) {
            offspring.clusters.row(i) = parent1.clusters.row(i);
            offspring.covariances[i] = parent1.covariances[i];
        } else {
            offspring.clusters.row(i) = parent2.clusters.row(assignment[i]);
            offspring.covariances[i] = parent2.covariances[assignment[i]];
        }
    }

    for (int i = 0; i < k; ++i) {
        offspring.weights[i] = (parent1.weights[i] + parent2.weights[assignment[i]]) / 2;
    }

    return offspring;
}


GeneticalAlgorithm::GeneticalAlgorithm(const std::mt19937 &rngRef, const int max_iterations,
                                       const int max_iterations_without_improvement,
                                       const int pop_min_size,
                                       const int pop_max_size,
                                       const bool verbose): rng(rngRef),
                                                            verbose(verbose),
                                                            max_iterations(max_iterations),
                                                            max_iterations_without_improvement(
                                                                max_iterations_without_improvement),
                                                            pop_min_size(pop_min_size),
                                                            pop_max_size(pop_max_size) {
}

GeneticalAlgorithm::GeneticalAlgorithm(const unsigned int seed, const int max_iterations,
                                       const int max_iterations_without_improvement,
                                       const int pop_min_size,
                                       const int pop_max_size,
                                       const bool verbose): rng(seed),
                                                            verbose(verbose),
                                                            max_iterations(max_iterations),
                                                            max_iterations_without_improvement(
                                                                max_iterations_without_improvement),
                                                            pop_min_size(pop_min_size),
                                                            pop_max_size(pop_max_size) {
}
