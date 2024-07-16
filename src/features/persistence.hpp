// src/features/persistence.hpp
#pragma once

#include "../../include/vector_database.hpp"
#include <string>

class Persistence {
public:
    static void save(const VectorDatabase& db, const std::string& filename);
    static VectorDatabase load(const std::string& filename);
};
