#ifndef GENETICALALGORITHM_H
#define GENETICALALGORITHM_H

#include "gmm/gmmResult.h"
#include <vector>
#include <set>
#include <random>

class GeneticalAlgorithm {
    std::vector<GMMResult> population;
    std::set<int> empty;

    [[nodiscard]] size_t population_size() const;

    [[nodiscard]] size_t active_population_size() const;

    void remove(size_t i);

    void add(const GMMResult &result);

    bool is_active(size_t i);

public:
    GeneticalAlgorithm() = default;
};


#endif // GENETICALALGORITHM_H
