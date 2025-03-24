#ifndef GENETICALALGORITHM_H
#define GENETICALALGORITHM_H

#include "gmm/gmmResult.h"
#include <vector>
#include <set>
#include <random>

class GeneticalAlgorithm {
    class Population {
        std::vector<GMMResult> population;
        std::set<int> empty;

    public:
        [[nodiscard]] size_t size() const {
            return population.size();
        }

        [[nodiscard]] size_t active_size() const {
            return population.size() - empty.size();
        }

        GMMResult &operator[](const size_t i) {
            return population[i];
        }

        void remove(const size_t i) {
            if (i < population.size()) {
                population[i].reset();
                empty.insert(i);
            }
        }

        void add(const GMMResult &result) {
            if (!empty.empty()) {
                const int idx = *empty.begin();
                empty.erase(empty.begin());
                population[idx] = result;
            } else {
                population.push_back(result);
            }
        }

        bool is_active(const size_t i) {
            return empty.find(i) == empty.end();
        }

        void eliminate(size_t to_remove, std::mt19937 &rng);
    };

    Population pop;
    std::mt19937 &rng;

    std::pair<GMMResult, GMMResult> binary_tournament();

    GMMResult get_best_solution();

    void mutate(GMMResult &individual, const Eigen::MatrixXd &data) const;

    [[nodiscard]] GMMResult crossover(const GMMResult &parent1, const GMMResult &parent2,
                                      const Eigen::MatrixXd &data) const;

public:
    explicit GeneticalAlgorithm(std::mt19937 &rngRef) : rng(rngRef) {
    }

    GMMResult run(const Eigen::MatrixXd &data);
};

#endif // GENETICALALGORITHM_H
