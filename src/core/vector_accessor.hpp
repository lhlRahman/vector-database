#pragma once

#include <cstdint>
#include <functional>

// Callback that maps a storage slot ID to a raw float pointer.
// Used by indexes to read vector data from mmap without copying.
using VectorAccessor = std::function<const float*(uint64_t)>;
