# Gaussian Mixture Models -- Regularized EM Algorithm

## Master Thesis Project

This repository contains the C++ implementation developed as part of my
master's thesis, focused on the performance and stability of Gaussian
Mixture Models (GMM) using several covariance regularization
techniques.\
The goal of the project is to compare the classical EM-based GMM with
multiple regularized variants and evaluate their behavior against a
reference implementation written in Julia.

The work investigates:

-   efficiency of different covariance regularization strategies,
-   stability of the EM algorithm under ill-conditioned covariance
    matrices,
-   computational performance relative to the Julia baseline,
-   clustering accuracy measured with ARI and NMI.

## Features

-   Implementation of the EM algorithm for Gaussian Mixture Models in
    C++.
-   Support for several covariance estimation methods:
    -   empirical covariance,
    -   Ledoit--Wolf shrinkage,
    -   OAS (Oracle Approximating Shrinkage),
    -   classical shrinkage estimators.
-   Numerical operations based on:
    -   Eigen (matrix algebra),
    -   OpenBLAS (BLAS acceleration).
-   Synthetic dataset generator compatible with experimental settings
    described in:
    -   Sampaio et al., Regularization and optimization in model-based
        clustering, Pattern Recognition, 2024.
-   Benchmarking tools for:
    -   time per EM iteration,
    -   ARI (Adjusted Rand Index),
    -   NMI (Normalized Mutual Information).

## Dependencies

The project requires:

-   C++17 or newer,
-   Eigen 3.4+,
-   OpenBLAS,
-   CMake (version 3.20 or newer).

Example installation on Arch Linux (WSL2):

``` bash
sudo pacman -S eigen openblas cmake gcc
```

## Building the Project

The project uses CMake.

### 1. Clone the repository

``` bash
git clone https://github.com/<your-name>/<your-repo>.git
cd <your-repo>
```

### 2. Configure the build

``` bash
cmake -B build   -G "Unix Makefiles"   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_C_COMPILER=/usr/bin/gcc   -DCMAKE_CXX_COMPILER=/usr/bin/g++   -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -pipe"   -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

### 3. Compile

``` bash
cmake --build build --config Release
```

### 4. Run

``` bash
./build/gmm_executable <config-file>
```

## Benchmarking Environment

All experiments were executed under a controlled, single-thread
environment to ensure fair comparisons.

### C++

``` bash
export OPENBLAS_NUM_THREADS=1
```

### Julia reference implementation

``` bash
JULIA_NUM_THREADS=1 julia --threads 1
```

## Evaluation Metrics

The project computes several performance and quality metrics:

-   Δ time per EM iteration (difference between C++ and Julia),
-   Δ ARI,
-   Δ NMI.

## Reference

Sampaio, R. A., Garcia, J. D., Poggi, M., Vidal, T. (2024).\
Regularization and optimization in model-based clustering.\
Pattern Recognition, 150.

## License

MIT License

