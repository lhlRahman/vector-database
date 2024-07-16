// src/features/dimensionality_reduction.hpp

#pragma once
#include "../core/vector.hpp"
#include <vector>

class PCA {
private:
    std::vector<std::vector<float>> components;
    std::vector<float> mean;
    int original_dim;
    int reduced_dim;

    // Helper functions
    std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) const;
    std::vector<float> multiply(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) const;
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix) const;
    std::vector<std::vector<float>> covariance_matrix(const std::vector<std::vector<float>>& data) const;
    
    // QR algorithm related functions
    void qr_decomposition(std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& Q, std::vector<std::vector<float>>& R) const;
    void qr_algorithm(std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& eigenvectors, std::vector<float>& eigenvalues) const;
    std::vector<std::vector<float>> multiply_matrices(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) const;

public:
    PCA(int n_components);
    void fit(const std::vector<Vector>& data);
    Vector transform(const Vector& v) const;
    Vector inverse_transform(const Vector& v) const;
};