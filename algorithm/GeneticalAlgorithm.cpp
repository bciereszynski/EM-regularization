//
// Created by bcier on 3/22/2025.
//

#include "GeneticalAlgorithm.h"


#include <algorithm>

GMMResult GeneticalAlgorithm::run() {
    auto best_obj = 0.0;
    auto iterations_without_improvement = 0;

    // initialize base population

    for (int iter = 0; iter < 100; ++iter) {
        auto [parent1, parent2] = binary_tournament();

        // auto child = crossover(parent1, parent2, rng);

        // mutate(child);

        // local search - as template
        // add(child);

        // eliminate worst

        // check for improvement
        const auto best_solution = get_best_solution();
        if (best_solution.objective > best_obj) {
            best_obj = best_solution.objective;
            iterations_without_improvement = 0;
        } else {
            ++iterations_without_improvement;
        }

        if (iterations_without_improvement > 10) {
            break;
        }
    }
    return get_best_solution();
}

void GeneticalAlgorithm::add(const GMMResult &result) {
    if (!empty.empty()) {
        const int idx = *empty.begin();
        empty.erase(empty.begin());
        population[idx] = result;
    } else {
        population.push_back(result);
    }
}

void GeneticalAlgorithm::remove(const size_t i) {
    if (i < population.size()) {
        population[i].reset();
        empty.insert(i);
    }
}

size_t GeneticalAlgorithm::active_population_size() const {
    return population.size() - empty.size();
}

size_t GeneticalAlgorithm::population_size() const {
    return population.size();
}

bool GeneticalAlgorithm::is_active(const size_t i) {
    return empty.find(i) != empty.end();
}

std::pair<GMMResult, GMMResult> GeneticalAlgorithm::binary_tournament() {
    std::vector<int> valid_indices(population_size());
    std::iota(valid_indices.begin(), valid_indices.end(), 0);

    valid_indices.erase(
        std::remove_if(valid_indices.begin(), valid_indices.end(),
                       [&](int i) { return is_active(i); }),
        valid_indices.end()
    );

    if (valid_indices.size() < 4) {
        throw std::runtime_error("Not enough active individuals for tournament selection.");
    }
    std::shuffle(valid_indices.begin(), valid_indices.end(), rng);

    const GMMResult &parent1 = population[valid_indices[0]];
    const GMMResult &parent2 = population[valid_indices[1]];
    const GMMResult &parent3 = population[valid_indices[2]];
    const GMMResult &parent4 = population[valid_indices[3]];

    return {
        is_better(parent1, parent2) ? parent1 : parent2,
        is_better(parent3, parent4) ? parent3 : parent4
    };
}

GMMResult GeneticalAlgorithm::get_best_solution() {
    if (active_population_size() == 0) {
        throw std::runtime_error("No active solutions in population.");
    }

    GMMResult best_solution = population[0];
    bool found = false;

    for (size_t i = 0; i < population_size(); ++i) {
        if (empty.find(i) == empty.end()) {
            if (!found || is_better(population[i], best_solution)) {
                best_solution = population[i];
                found = true;
            }
        }
    }

    return best_solution;
}
