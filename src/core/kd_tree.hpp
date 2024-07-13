#pragma once
#include "vector.hpp"
#include <memory>
#include <string>

class KDTree {
    private:
        struct Node {
            Vector vector;
            std::string key;
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            int split_dimension;

            Node(const Vector& vec, const std::string& k) : vector(vec), key(k), split_dimension(-1) {}
        };

        std::unique_ptr<Node> root;
        size_t dimensions;

    void insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, int depth);
    void nearest_neighbor_recursive(const Node* node, const Vector& query, std::string& best_key, float& best_distance, int depth) const;

    public:
        KDTree(size_t dimensions);
        void insert(const Vector& vector, const std::string& key);
        std::string nearest_neighbor(const Vector& query) const;

};