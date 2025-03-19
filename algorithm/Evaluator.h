#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <vector>
#include "gmm/gmmResult.h"

class Evaluator {
    std::vector<std::vector<double> > data;
    std::vector<int> expected_clusters;
    int k;
    GMMResult result;

public:
    Evaluator(const std::vector<std::vector<double> > &data, const std::vector<int> &expected_clusters, int k,
              GMMResult result);

    int centroid_index_sphere();
};


#endif //EVALUATOR_H
