#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include "gpu_operations.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>

namespace {
    // Metal objects (global state)
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;
    
    // Compiled pipeline states (kernels)
    id<MTLComputePipelineState> batchDotProductsPipeline = nil;
    id<MTLComputePipelineState> batchEuclideanPipeline = nil;
    id<MTLComputePipelineState> batchCosinePipeline = nil;
    
    bool initialized = false;
    
    // ==================== Persistent Database Buffer ====================
    // This is the key optimization - we keep the database in GPU-accessible memory
    // and only update it when data changes, not on every search!
    
    id<MTLBuffer> databaseBuffer = nil;      // Persistent buffer for database vectors
    id<MTLBuffer> resultsBuffer = nil;       // Reusable results buffer
    size_t bufferNumVectors = 0;
    size_t bufferDimensions = 0;
    size_t resultsBufferSize = 0;
    
    // Helper: Create pipeline from function name
    id<MTLComputePipelineState> createPipeline(NSString* functionName) {
        id<MTLFunction> function = [library newFunctionWithName:functionName];
        if (!function) {
            NSLog(@"Failed to find function: %@", functionName);
            return nil;
        }
        
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = 
            [device newComputePipelineStateWithFunction:function error:&error];
        
        if (error) {
            NSLog(@"Failed to create pipeline for %@: %@", functionName, error);
            return nil;
        }
        
        return pipeline;
    }
    
    // Helper: Create buffer with shared storage (zero-copy on Apple Silicon!)
    id<MTLBuffer> createSharedBuffer(size_t byteSize) {
        return [device newBufferWithLength:byteSize 
                                   options:MTLResourceStorageModeShared];
    }
    
    // Helper: Create buffer from data with shared storage
    id<MTLBuffer> createSharedBuffer(const void* data, size_t byteSize) {
        return [device newBufferWithBytes:data 
                                   length:byteSize
                                  options:MTLResourceStorageModeShared];
    }
    
    // Helper: Ensure results buffer is large enough
    void ensureResultsBuffer(size_t numVectors) {
        size_t needed = numVectors * sizeof(float);
        if (resultsBuffer == nil || resultsBufferSize < needed) {
            resultsBuffer = createSharedBuffer(needed);
            resultsBufferSize = needed;
        }
    }
}

namespace gpu_ops {

// ==================== Lifecycle ====================

bool initialize() {
    if (initialized) return true;
    
    @autoreleasepool {
        // 1. Get the default Metal device (GPU)
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return false;
        }
        
        NSLog(@"Using GPU: %@", device.name);
        NSLog(@"Unified Memory: %@", device.hasUnifiedMemory ? @"YES (zero-copy enabled)" : @"NO");
        
        // 2. Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            NSLog(@"Failed to create command queue");
            return false;
        }
        
        // 3. Load shader library - compile from source at runtime
        NSString* shaderPath = @"src/optimizations/shaders/vector_ops.metal";
        NSString* source = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil];
        if (source) {
            NSError* error = nil;
            library = [device newLibraryWithSource:source options:nil error:&error];
            if (error) {
                NSLog(@"Failed to compile shaders: %@", error);
                return false;
            }
        } else {
            NSLog(@"Failed to load shader source from %@", shaderPath);
            return false;
        }
        
        // 4. Create pipeline states for each kernel
        batchDotProductsPipeline = createPipeline(@"batch_dot_products");
        batchEuclideanPipeline = createPipeline(@"batch_euclidean_distances");
        batchCosinePipeline = createPipeline(@"batch_cosine_distances");
        
        if (!batchEuclideanPipeline) {
            NSLog(@"Failed to create euclidean pipeline - GPU search won't work");
            return false;
        }
        
        initialized = true;
        NSLog(@"GPU acceleration initialized successfully");
        return true;
    }
}

bool shutdown() {
    @autoreleasepool {
        clear_database_buffer();
        batchDotProductsPipeline = nil;
        batchEuclideanPipeline = nil;
        batchCosinePipeline = nil;
        library = nil;
        commandQueue = nil;
        device = nil;
        initialized = false;
        return true;
    }
}

bool is_available() {
    if (!initialized) {
        return initialize();
    }
    return device != nil && batchEuclideanPipeline != nil;
}

// ==================== Database Buffer Management ====================

bool set_database_buffer(const float* data, size_t num_vectors, size_t dimensions) {
    if (!is_available()) return false;
    if (data == nullptr || num_vectors == 0 || dimensions == 0) return false;
    
    @autoreleasepool {
        size_t byteSize = num_vectors * dimensions * sizeof(float);
        
        // Create new buffer with the data
        // MTLResourceStorageModeShared means CPU and GPU share the same memory!
        databaseBuffer = [device newBufferWithBytes:data
                                             length:byteSize
                                            options:MTLResourceStorageModeShared];
        
        if (!databaseBuffer) {
            NSLog(@"Failed to create database buffer");
            return false;
        }
        
        bufferNumVectors = num_vectors;
        bufferDimensions = dimensions;
        
        // Pre-allocate results buffer
        ensureResultsBuffer(num_vectors);
        
        NSLog(@"GPU database buffer set: %zu vectors x %zu dims = %.2f MB", 
              num_vectors, dimensions, byteSize / (1024.0 * 1024.0));
        
        return true;
    }
}

bool update_database_buffer(const float* data, size_t num_vectors, size_t dimensions) {
    // For now, just recreate the buffer
    // A more sophisticated implementation could update in-place if size matches
    return set_database_buffer(data, num_vectors, dimensions);
}

void clear_database_buffer() {
    @autoreleasepool {
        databaseBuffer = nil;
        resultsBuffer = nil;
        bufferNumVectors = 0;
        bufferDimensions = 0;
        resultsBufferSize = 0;
    }
}

bool has_database_buffer() {
    return databaseBuffer != nil && bufferNumVectors > 0;
}

size_t get_buffer_num_vectors() {
    return bufferNumVectors;
}

size_t get_buffer_dimensions() {
    return bufferDimensions;
}

// ==================== Search Operations (zero-copy) ====================

std::vector<float> search_euclidean(const Vector& query) {
    if (!is_available() || !has_database_buffer()) {
        std::cerr << "[GPU] search_euclidean: not available or no buffer" << std::endl;
        return {};
    }
    
    if (query.size() != bufferDimensions) {
        std::cerr << "[GPU] Query dimension mismatch: " << query.size() 
                  << " vs buffer " << bufferDimensions << std::endl;
        return {};
    }
    
    @autoreleasepool {
        // Create query buffer (small, so copy is fine)
        id<MTLBuffer> queryBuffer = createSharedBuffer(query.data_ptr(), 
                                                        bufferDimensions * sizeof(float));
        
        // Ensure results buffer is ready
        ensureResultsBuffer(bufferNumVectors);
        
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline and buffers
        [encoder setComputePipelineState:batchEuclideanPipeline];
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:databaseBuffer offset:0 atIndex:1];  // Persistent buffer!
        [encoder setBuffer:resultsBuffer offset:0 atIndex:2];
        
        uint32_t dims = static_cast<uint32_t>(bufferDimensions);
        uint32_t numVecs = static_cast<uint32_t>(bufferNumVectors);
        [encoder setBytes:&dims length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&numVecs length:sizeof(uint32_t) atIndex:4];
        
        // Calculate thread groups
        NSUInteger threadGroupSize = batchEuclideanPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > bufferNumVectors) {
            threadGroupSize = bufferNumVectors;
        }
        
        [encoder dispatchThreads:MTLSizeMake(bufferNumVectors, 1, 1) 
           threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Read results directly from shared memory (no copy on Apple Silicon!)
        std::vector<float> results(bufferNumVectors);
        float* resultPtr = static_cast<float*>([resultsBuffer contents]);
        std::copy(resultPtr, resultPtr + bufferNumVectors, results.begin());
        
        return results;
    }
}

std::vector<float> search_dot_product(const Vector& query) {
    if (!is_available() || !has_database_buffer()) {
        return {};
    }
    
    if (query.size() != bufferDimensions) {
        return {};
    }
    
    @autoreleasepool {
        id<MTLBuffer> queryBuffer = createSharedBuffer(query.data_ptr(), 
                                                        bufferDimensions * sizeof(float));
        ensureResultsBuffer(bufferNumVectors);
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:batchDotProductsPipeline];
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:databaseBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultsBuffer offset:0 atIndex:2];
        
        uint32_t dims = static_cast<uint32_t>(bufferDimensions);
        uint32_t numVecs = static_cast<uint32_t>(bufferNumVectors);
        [encoder setBytes:&dims length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&numVecs length:sizeof(uint32_t) atIndex:4];
        
        NSUInteger threadGroupSize = batchDotProductsPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > bufferNumVectors) threadGroupSize = bufferNumVectors;
        
        [encoder dispatchThreads:MTLSizeMake(bufferNumVectors, 1, 1) 
           threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::vector<float> results(bufferNumVectors);
        float* resultPtr = static_cast<float*>([resultsBuffer contents]);
        std::copy(resultPtr, resultPtr + bufferNumVectors, results.begin());
        
        return results;
    }
}

std::vector<float> search_cosine(const Vector& query) {
    if (!is_available() || !has_database_buffer() || !batchCosinePipeline) {
        return {};
    }
    
    if (query.size() != bufferDimensions) {
        return {};
    }
    
    @autoreleasepool {
        id<MTLBuffer> queryBuffer = createSharedBuffer(query.data_ptr(), 
                                                        bufferDimensions * sizeof(float));
        ensureResultsBuffer(bufferNumVectors);
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:batchCosinePipeline];
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:databaseBuffer offset:0 atIndex:1];
        [encoder setBuffer:resultsBuffer offset:0 atIndex:2];
        
        uint32_t dims = static_cast<uint32_t>(bufferDimensions);
        uint32_t numVecs = static_cast<uint32_t>(bufferNumVectors);
        [encoder setBytes:&dims length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&numVecs length:sizeof(uint32_t) atIndex:4];
        
        NSUInteger threadGroupSize = batchCosinePipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > bufferNumVectors) threadGroupSize = bufferNumVectors;
        
        [encoder dispatchThreads:MTLSizeMake(bufferNumVectors, 1, 1) 
           threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::vector<float> results(bufferNumVectors);
        float* resultPtr = static_cast<float*>([resultsBuffer contents]);
        std::copy(resultPtr, resultPtr + bufferNumVectors, results.begin());
        
        return results;
    }
}

// ==================== Legacy API (copies data each call) ====================

std::vector<float> batch_dot_products(
    const Vector& query,
    const std::vector<Vector>& database
) {
    if (!is_available() || database.empty()) {
        return {};
    }
    
    @autoreleasepool {
        size_t dimensions = query.size();
        size_t num_vectors = database.size();
        
        // Flatten database vectors into contiguous array
        std::vector<float> flat_database(num_vectors * dimensions);
        for (size_t i = 0; i < num_vectors; i++) {
            std::copy(database[i].begin(), database[i].end(), 
                      flat_database.begin() + i * dimensions);
        }
        
        // Create Metal buffers
        id<MTLBuffer> queryBuffer = createSharedBuffer(query.data_ptr(), dimensions * sizeof(float));
        id<MTLBuffer> databaseBuf = createSharedBuffer(flat_database.data(), flat_database.size() * sizeof(float));
        id<MTLBuffer> resultsBuf = createSharedBuffer(num_vectors * sizeof(float));
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:batchDotProductsPipeline];
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:databaseBuf offset:0 atIndex:1];
        [encoder setBuffer:resultsBuf offset:0 atIndex:2];
        
        uint32_t dims = static_cast<uint32_t>(dimensions);
        uint32_t numVecs = static_cast<uint32_t>(num_vectors);
        [encoder setBytes:&dims length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&numVecs length:sizeof(uint32_t) atIndex:4];
        
        NSUInteger threadGroupSize = batchDotProductsPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > num_vectors) threadGroupSize = num_vectors;
        
        [encoder dispatchThreads:MTLSizeMake(num_vectors, 1, 1) 
           threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::vector<float> results(num_vectors);
        float* resultPtr = static_cast<float*>([resultsBuf contents]);
        std::copy(resultPtr, resultPtr + num_vectors, results.begin());
        
        return results;
    }
}

std::vector<float> batch_euclidean_distances(
    const Vector& query,
    const std::vector<Vector>& database
) {
    if (!is_available() || database.empty()) {
        return {};
    }
    
    @autoreleasepool {
        size_t dimensions = query.size();
        size_t num_vectors = database.size();
        
        std::vector<float> flat_database(num_vectors * dimensions);
        for (size_t i = 0; i < num_vectors; i++) {
            std::copy(database[i].begin(), database[i].end(), 
                      flat_database.begin() + i * dimensions);
        }
        
        id<MTLBuffer> queryBuffer = createSharedBuffer(query.data_ptr(), dimensions * sizeof(float));
        id<MTLBuffer> databaseBuf = createSharedBuffer(flat_database.data(), flat_database.size() * sizeof(float));
        id<MTLBuffer> resultsBuf = createSharedBuffer(num_vectors * sizeof(float));
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:batchEuclideanPipeline];
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:databaseBuf offset:0 atIndex:1];
        [encoder setBuffer:resultsBuf offset:0 atIndex:2];
        
        uint32_t dims = static_cast<uint32_t>(dimensions);
        uint32_t numVecs = static_cast<uint32_t>(num_vectors);
        [encoder setBytes:&dims length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&numVecs length:sizeof(uint32_t) atIndex:4];
        
        NSUInteger threadGroupSize = batchEuclideanPipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > num_vectors) threadGroupSize = num_vectors;
        
        [encoder dispatchThreads:MTLSizeMake(num_vectors, 1, 1) 
           threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        std::vector<float> results(num_vectors);
        float* resultPtr = static_cast<float*>([resultsBuf contents]);
        std::copy(resultPtr, resultPtr + num_vectors, results.begin());
        
        return results;
    }
}

std::vector<std::vector<std::pair<size_t, float>>> batch_knn(
    const std::vector<Vector>& queries,
    const std::vector<Vector>& database,
    size_t k
) {
    std::vector<std::vector<std::pair<size_t, float>>> results;
    results.reserve(queries.size());
    
    for (const auto& query : queries) {
        auto distances = batch_euclidean_distances(query, database);
        
        std::vector<std::pair<size_t, float>> indexed_distances;
        indexed_distances.reserve(distances.size());
        for (size_t i = 0; i < distances.size(); i++) {
            indexed_distances.emplace_back(i, distances[i]);
        }
        
        std::partial_sort(indexed_distances.begin(),
                          indexed_distances.begin() + std::min(k, indexed_distances.size()),
                          indexed_distances.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
        
        indexed_distances.resize(std::min(k, indexed_distances.size()));
        results.push_back(std::move(indexed_distances));
    }
    
    return results;
}

} // namespace gpu_ops
