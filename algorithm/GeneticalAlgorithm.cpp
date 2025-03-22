//
// Created by bcier on 3/22/2025.
//

#include "GeneticalAlgorithm.h"


#include <algorithm>

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
