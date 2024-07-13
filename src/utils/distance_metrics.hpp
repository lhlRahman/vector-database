#pragma once
# include "../core/vector.hpp"

namespace DistanceMetrics {
    float euclidean (const Vector& v1, const Vector& v2);
    float manhattan (const Vector& v1, const Vector& v2);
    float cosine (const Vector& v1, const Vector& v2);
}
