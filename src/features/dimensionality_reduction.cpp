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

    // Convert data to 2D vector and compute mean
    std::vector<std::vector<float>> X(n_samples, std::vector<float>(original_dim));
    mean.resize(original_dim, 0.0f);
    
    // Compute mean and center the data in one pass
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            X[i][j] = data[i][j];
            mean[j] += X[i][j];
        }
    }
    
    // Finalize mean computation
    for (int j = 0; j < original_dim; ++j) {
        mean[j] /= n_samples;
        // Center the data
        for (int i = 0; i < n_samples; ++i) {
            X[i][j] -= mean[j];
        }
    }

    // Compute covariance matrix with better numerical stability
    auto cov = compute_covariance_matrix(X, n_samples);
    
    std::cout << "Computing eigendecomposition..." << std::endl;
    
    // Initialize eigenvectors as identity matrix
    components.resize(original_dim, std::vector<float>(original_dim, 0.0f));
    for (int i = 0; i < original_dim; ++i) {
        components[i][i] = 1.0f;
    }
    
    // Power iteration method for finding principal components
    std::vector<float> eigenvalues(original_dim);
    for (int i = 0; i < original_dim; ++i) {
        std::vector<float> eigenvector = power_iteration(cov, 100);
        eigenvalues[i] = compute_rayleigh_quotient(cov, eigenvector);
        
        // Deflate the matrix
        deflate_matrix(cov, eigenvector, eigenvalues[i]);
        
        // Store the eigenvector
        for (int j = 0; j < original_dim; ++j) {
            components[i][j] = eigenvector[j];
        }
    }

    // Sort eigenvectors by eigenvalues
    std::vector<size_t> indices(original_dim);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&eigenvalues](size_t i1, size_t i2) { return eigenvalues[i1] > eigenvalues[i2]; });

    // Select top k eigenvectors
    std::vector<std::vector<float>> temp_components(reduced_dim, std::vector<float>(original_dim));
    for (int i = 0; i < reduced_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            temp_components[i][j] = components[indices[i]][j];
        }
    }
    components = temp_components;

    std::cout << "PCA fit completed. Components shape: " << components.size() << "x" << components[0].size() << std::endl;
}

std::vector<std::vector<float>> PCA::compute_covariance_matrix(const std::vector<std::vector<float>>& X, int n_samples) const {
    int n = X[0].size();
    std::vector<std::vector<float>> cov(n, std::vector<float>(n, 0.0f));
    
    float scale = 1.0f / (n_samples - 1);
    
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float sum = 0.0f;
            for (const auto& sample : X) {
                sum += sample[i] * sample[j];
            }
            cov[i][j] = sum * scale;
            if (i != j) {
                cov[j][i] = cov[i][j];  // Matrix is symmetric
            }
        }
    }
    return cov;
}

std::vector<float> PCA::power_iteration(const std::vector<std::vector<float>>& matrix, int num_iterations) const {
    int n = matrix.size();
    std::vector<float> vector(n, 1.0f);
    
    // Normalize initial vector
    float norm = 0.0f;
    for (float val : vector) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : vector) {
        val /= norm;
    }
    
    // Power iteration
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<float> new_vector(n, 0.0f);
        
        // Matrix-vector multiplication
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                new_vector[i] += matrix[i][j] * vector[j];
            }
        }
        
        // Normalize
        norm = 0.0f;
        for (float val : new_vector) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        for (float& val : new_vector) {
            val /= norm;
        }
        
        vector = new_vector;
    }
    
    return vector;
}

float PCA::compute_rayleigh_quotient(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vector) const {
    std::vector<float> Av(vector.size(), 0.0f);
    
    // Compute A*v
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            Av[i] += matrix[i][j] * vector[j];
        }
    }
    
    // Compute v^T * A * v
    float numerator = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        numerator += vector[i] * Av[i];
    }
    
    // Compute v^T * v
    float denominator = 0.0f;
    for (float val : vector) {
        denominator += val * val;
    }
    
    return numerator / denominator;
}

void PCA::deflate_matrix(std::vector<std::vector<float>>& matrix, const std::vector<float>& eigenvector, float eigenvalue) const {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
}

Vector PCA::transform(const Vector& v) const {
    if (v.size() != original_dim) {
        throw std::invalid_argument("Vector dimension does not match the original dimension");
    }
    
    std::cout << "Transforming vector of size " << v.size() << std::endl;

    // Center the data
    std::vector<float> centered(original_dim);
    for (size_t i = 0; i < original_dim; ++i) {
        centered[i] = v[i] - mean[i];
    }

    // Project onto principal components
    std::vector<float> transformed(reduced_dim, 0.0f);
    for (size_t i = 0; i < reduced_dim; ++i) {
        for (size_t j = 0; j < original_dim; ++j) {
            transformed[i] += components[i][j] * centered[j];
        }
    }

    return Vector(transformed);
}

Vector PCA::inverse_transform(const Vector& v) const {
    if (v.size() != reduced_dim) {
        throw std::invalid_argument("Vector dimension does not match the reduced dimension");
    }
    
    std::vector<float> reconstructed(original_dim, 0.0f);
    
    // Project back to original space
    for (size_t i = 0; i < original_dim; ++i) {
        for (size_t j = 0; j < reduced_dim; ++j) {
            reconstructed[i] += components[j][i] * v[j];
        }
        // Add back the mean
        reconstructed[i] += mean[i];
    }
    
    return Vector(reconstructed);
}