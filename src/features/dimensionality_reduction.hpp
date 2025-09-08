#pragma once

#include <algorithm> // Added for std::transform
#include <vector>

#include "../core/vector.hpp"

class PCA
{
private:
    std::vector<std::vector<float>> components;
    std::vector<float> mean;
    int original_dim;
    int reduced_dim;

    // Original helper functions
    std::vector<float> subtract(const std::vector<float> &a, const std::vector<float> &b) const;
    std::vector<float> multiply(const std::vector<std::vector<float>> &matrix, const std::vector<float> &vec) const;
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix) const;
    std::vector<std::vector<float>> covariance_matrix(const std::vector<std::vector<float>> &data) const;

    // Improved numerical methods
    std::vector<std::vector<float>> compute_covariance_matrix(
        const std::vector<std::vector<float>> &X,
        int n_samples) const;

    std::vector<float> power_iteration(
        const std::vector<std::vector<float>> &matrix,
        int num_iterations) const;

    float compute_rayleigh_quotient(
        const std::vector<std::vector<float>> &matrix,
        const std::vector<float> &vector) const;

    void deflate_matrix(
        std::vector<std::vector<float>> &matrix,
        const std::vector<float> &eigenvector,
        float eigenvalue) const;

    // QR algorithm as fallback
    void qr_decomposition(
        std::vector<std::vector<float>> &A,
        std::vector<std::vector<float>> &Q,
        std::vector<std::vector<float>> &R) const;

    void qr_algorithm(
        std::vector<std::vector<float>> &A,
        std::vector<std::vector<float>> &eigenvectors,
        std::vector<float> &eigenvalues) const;

    std::vector<std::vector<float>> multiply_matrices(
        const std::vector<std::vector<float>> &A,
        const std::vector<std::vector<float>> &B) const;

public:
    explicit PCA(int n_components);
    void fit(const std::vector<Vector> &data);
    Vector transform(const Vector &v) const;
    Vector inverse_transform(const Vector &v) const;
};