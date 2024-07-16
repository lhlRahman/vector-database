// src/features/dimensionality_reduction.cpp
#include "dimensionality_reduction.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>

PCA::PCA(int n_components) : reduced_dim(n_components) {}

void PCA::fit(const std::vector<Vector>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data set is empty");
    }
    
    original_dim = data[0].size();
    int n_samples = data.size();

    std::cout << "Fitting PCA with " << n_samples << " samples of dimension " << original_dim << std::endl;

    // Convert data to 2D vector
    std::vector<std::vector<float>> X(n_samples, std::vector<float>(original_dim));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            X[i][j] = data[i][j];
        }
    }

    // Compute mean
    mean.resize(original_dim, 0.0f);
    for (int j = 0; j < original_dim; ++j) {
        for (int i = 0; i < n_samples; ++i) {
            mean[j] += X[i][j];
        }
        mean[j] /= n_samples;
    }

    // Center the data
    for (int i = 0; i < n_samples; ++i) {
        X[i] = subtract(X[i], mean);
    }

    // Compute covariance matrix
    auto cov = covariance_matrix(X);

    std::cout << "Covariance matrix computed. Size: " << cov.size() << "x" << cov[0].size() << std::endl;

    // Compute eigenvectors and eigenvalues using QR algorithm
    components.resize(original_dim, std::vector<float>(original_dim));
    std::vector<float> eigenvalues(original_dim);
    qr_algorithm(cov, components, eigenvalues);

    std::cout << "Eigenvectors and eigenvalues computed." << std::endl;
    std::cout << "First few eigenvalues: ";
    for (size_t i = 0; i < std::min(size_t(5), eigenvalues.size()); ++i) {
        std::cout << eigenvalues[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Sort eigenvectors by eigenvalues in descending order
    std::vector<size_t> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&eigenvalues](size_t i1, size_t i2) { return eigenvalues[i1] > eigenvalues[i2]; });

    // Select top k eigenvectors
    std::vector<std::vector<float>> temp_components(reduced_dim, std::vector<float>(original_dim));
    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            temp_components[i][j] = components[j][indices[i]];
        }
    }
    components = temp_components;

    std::cout << "PCA fit completed. Components shape: " << components.size() << "x" << components[0].size() << std::endl;
}

Vector PCA::transform(const Vector& v) const {
    if (v.size() != original_dim) {
        throw std::invalid_argument("Vector dimension does not match the original dimension");
    }
    
    std::cout << "Transforming vector of size " << v.size() << std::endl;

    std::vector<float> centered = subtract(std::vector<float>(v.data_ptr(), v.data_ptr() + v.size()), mean);
    std::vector<float> transformed(reduced_dim, 0.0f);

    for (size_t i = 0; i < reduced_dim; ++i) {
        for (size_t j = 0; j < original_dim; ++j) {
            transformed[i] += components[i][j] * centered[j];
        }
        if (std::isnan(transformed[i])) {
            std::cout << "NaN detected at index " << i << std::endl;
            std::cout << "Component vector: ";
            for (size_t k = 0; k < std::min(size_t(5), components[i].size()); ++k) {
                std::cout << components[i][k] << " ";
            }
            std::cout << "..." << std::endl;
            throw std::runtime_error("PCA transformation produced NaN values");
        }
    }

    std::cout << "Transformed vector: ";
    for (size_t i = 0; i < std::min(transformed.size(), size_t(5)); ++i) {
        std::cout << transformed[i] << " ";
    }
    std::cout << "..." << std::endl;

    return Vector(transformed);
}

Vector PCA::inverse_transform(const Vector& v) const {
    if (v.size() != reduced_dim) {
        throw std::invalid_argument("Vector dimension does not match the reduced dimension");
    }
    
    std::vector<float> reconstructed = multiply(transpose(components), std::vector<float>(v.data_ptr(), v.data_ptr() + v.size()));
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        reconstructed[i] += mean[i];
    }
    return Vector(reconstructed);
}

// Helper function implementations
std::vector<float> PCA::subtract(const std::vector<float>& a, const std::vector<float>& b) const {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

std::vector<float> PCA::multiply(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) const {
    std::vector<float> result(matrix.size(), 0.0f);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<std::vector<float>> PCA::transpose(const std::vector<std::vector<float>>& matrix) const {
    std::vector<std::vector<float>> result(matrix[0].size(), std::vector<float>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<float>> PCA::covariance_matrix(const std::vector<std::vector<float>>& data) const {
    int n = data.size();
    int m = data[0].size();
    std::vector<std::vector<float>> cov(m, std::vector<float>(m, 0.0f));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= n - 1;
        }
    }
    return cov;
}

void PCA::qr_decomposition(std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& Q, std::vector<std::vector<float>>& R) const {
    int n = A.size();
    Q = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
    R = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));

    for (int j = 0; j < n; ++j) {
        std::vector<float> v = A[j];
        
        for (int i = 0; i < j; ++i) {
            R[i][j] = 0.0f;
            for (int k = 0; k < n; ++k) {
                R[i][j] += Q[k][i] * A[k][j];
            }
            for (int k = 0; k < n; ++k) {
                v[k] -= R[i][j] * Q[k][i];
            }
        }
        
        float norm = 0.0f;
        for (int i = 0; i < n; ++i) {
            norm += v[i] * v[i];
        }
        norm = std::sqrt(norm);
        
        R[j][j] = norm;
        for (int i = 0; i < n; ++i) {
            Q[i][j] = v[i] / norm;
        }
    }
}

void PCA::qr_algorithm(std::vector<std::vector<float>>& A, std::vector<std::vector<float>>& eigenvectors, std::vector<float>& eigenvalues) const {
    int n = A.size();
    eigenvectors = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0f;
    }

    const int max_iterations = 1000;  // Increase max iterations
    const float epsilon = 1e-10;

    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<std::vector<float>> Q, R;
        qr_decomposition(A, Q, R);

        std::vector<std::vector<float>> new_A = multiply_matrices(R, Q);
        std::vector<std::vector<float>> new_eigenvectors = multiply_matrices(eigenvectors, Q);

        float diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                diff += std::abs(new_A[i][j] - A[i][j]);
            }
        }

        A = new_A;
        eigenvectors = new_eigenvectors;

        if (diff < epsilon) {
            std::cout << "QR algorithm converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }

        if (iter == max_iterations - 1) {
            std::cout << "Warning: QR algorithm did not converge after " << max_iterations << " iterations." << std::endl;
        }
    }

    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A[i][i];
    }
}

std::vector<std::vector<float>> PCA::multiply_matrices(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) const {
    int m = A.size();
    int n = B[0].size();
    int p = A[0].size();
    std::vector<std::vector<float>> result(m, std::vector<float>(n, 0.0f));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}