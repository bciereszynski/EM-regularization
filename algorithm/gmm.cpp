#include "GMM.h"
#include <iostream>

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
    decompose_if_fails(decompose_if_fails) {}

void GMM::fit() {
    if (verbose) {
        std::cout << "Starting GMM...\n";
    }
}
