#ifndef GENETICALALGORITHM_H
#define GENETICALALGORITHM_H

#include "gmm/gmmResult.h"
#include <vector>
#include <set>
#include <random>

constexpr int GA_DEFAULT_MAX_ITERATIONS = 200;
constexpr int GA_DEFAULT_MAX_ITERATIONS_WITHOUT_IMPROVEMENT = 150;
constexpr int GA_DEFAULT_POPULATION_MAX_SIZE = 50;
constexpr int GA_DEFAULT_POPULATION_MIN_SIZE = 40;

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

        void clear() {
            empty.clear();
            population.clear();
        }
    };

    Population pop;
    std::mt19937 rng;

    bool verbose;
    int max_iterations;
    int max_iterations_without_improvement;

    int pop_min_size;
    int pop_max_size;

    std::pair<GMMResult, GMMResult> binary_tournament();

    static double distance(const GMMResult &a, int i, const GMMResult &b, int j);

    GMMResult get_best_solution();

    void mutate(GMMResult &individual, const Eigen::MatrixXd &data);

    [[nodiscard]] GMMResult crossover(const GMMResult &parent1, const GMMResult &parent2,
                                      const Eigen::MatrixXd &data);

public:
    explicit GeneticalAlgorithm(const std::mt19937 &rngRef, int max_iterations = GA_DEFAULT_MAX_ITERATIONS,
                                int max_iterations_without_improvement =
                                        GA_DEFAULT_MAX_ITERATIONS_WITHOUT_IMPROVEMENT,
                                int pop_min_size = GA_DEFAULT_POPULATION_MIN_SIZE,
                                int pop_max_size = GA_DEFAULT_POPULATION_MAX_SIZE,
                                bool verbose = false);


    explicit GeneticalAlgorithm(unsigned int seed, int max_iterations = GA_DEFAULT_MAX_ITERATIONS,
                                int max_iterations_without_improvement =
                                        GA_DEFAULT_MAX_ITERATIONS_WITHOUT_IMPROVEMENT,
                                int pop_min_size = GA_DEFAULT_POPULATION_MIN_SIZE,
                                int pop_max_size = GA_DEFAULT_POPULATION_MAX_SIZE,
                                bool verbose = false);


    GMMResult run(const Eigen::MatrixXd &data, int k);

    GMMResult run(const std::vector<std::vector<double> > &data, int k);

    friend class GATest_FriendAccess;
};

#endif // GENETICALALGORITHM_H
