#ifndef GENETICALALGORITHM_H
#define GENETICALALGORITHM_H

#include "gmm/gmmResult.h"
#include <vector>
#include <set>
#include <random>

class GeneticalAlgorithm {
    std::vector<GMMResult> population;
    std::set<int> empty;
    std::mt19937 &rng;

    [[nodiscard]] size_t population_size() const;

    [[nodiscard]] size_t active_population_size() const;

    void remove(size_t i);

    void add(const GMMResult &result);

    bool is_active(size_t i);

    std::pair<GMMResult, GMMResult> binary_tournament();

    GMMResult get_best_solution();

public:
    GeneticalAlgorithm() = default;

    GMMResult run();
};


#endif // GENETICALALGORITHM_H
