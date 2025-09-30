#!/usr/bin/env python3
"""
Test script for SIMD operations via API
Tests enabling/disabling SIMD and performance comparison
"""

import requests
import json
import time
import random
from typing import List

# Configuration
BASE_URL = "http://localhost:8080"
DIMENSIONS = 128

def check_simd_status():
    """Check current SIMD status"""
    response = requests.get(f"{BASE_URL}/config/simd")
    response.raise_for_status()
    return response.json()

def toggle_simd(enabled: bool):
    """Enable or disable SIMD"""
    response = requests.put(
        f"{BASE_URL}/config/simd",
        json={"enabled": enabled}
    )
    response.raise_for_status()
    return response.json()

def insert_vector(key: str, vector: List[float]):
    """Insert a vector"""
    response = requests.post(
        f"{BASE_URL}/vectors",
        json={"key": key, "vector": vector}
    )
    return response.status_code == 200

def search(query: List[float], k: int = 10):
    """Perform similarity search"""
    response = requests.post(
        f"{BASE_URL}/search",
        json={"query": query, "k": k}
    )
    response.raise_for_status()
    return response.json()

def generate_random_vector(dimensions: int = DIMENSIONS) -> List[float]:
    """Generate a random vector"""
    return [random.random() for _ in range(dimensions)]

def benchmark_search(num_queries: int = 100, k: int = 10) -> float:
    """Benchmark search performance"""
    queries = [generate_random_vector() for _ in range(num_queries)]
    
    start = time.time()
    for query in queries:
        search(query, k)
    elapsed = time.time() - start
    
    return elapsed

def main():
    print("\n" + "="*70)
    print("🚀 SIMD OPERATIONS TEST SUITE")
    print("="*70)
    
    # Check server health
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not healthy!")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running! Start it with: ./build/vector_db_server")
        return
    
    print("✅ Server is running\n")
    
    # Check initial SIMD status
    print("="*70)
    print("📊 INITIAL SIMD STATUS")
    print("="*70)
    status = check_simd_status()
    print(f"SIMD Enabled: {status['simd_enabled']}")
    print(f"SIMD Type: {status['simd_type']}")
    print(f"SIMD Width: {status['simd_width']} floats per operation")
    print(f"SIMD Available: {status['simd_available']}")
    
    # Insert test vectors
    print("\n" + "="*70)
    print("📝 INSERTING TEST VECTORS")
    print("="*70)
    num_vectors = 1000
    print(f"Inserting {num_vectors} vectors...")
    for i in range(num_vectors):
        vector = generate_random_vector()
        insert_vector(f"simd_test_{i}", vector)
        if (i + 1) % 250 == 0:
            print(f"  Inserted {i + 1}/{num_vectors} vectors...")
    print(f"✅ Inserted {num_vectors} vectors")
    
    # Benchmark with SIMD enabled
    print("\n" + "="*70)
    print("⏱️  BENCHMARK: SIMD ENABLED")
    print("="*70)
    toggle_simd(True)
    print("✅ SIMD enabled")
    
    num_queries = 100
    print(f"Running {num_queries} search queries...")
    simd_time = benchmark_search(num_queries, k=10)
    print(f"SIMD time: {simd_time:.3f} seconds")
    print(f"Average per query: {simd_time/num_queries*1000:.2f}ms")
    
    # Benchmark with SIMD disabled (scalar fallback)
    print("\n" + "="*70)
    print("⏱️  BENCHMARK: SIMD DISABLED (Scalar Fallback)")
    print("="*70)
    toggle_simd(False)
    print("✅ SIMD disabled (using scalar operations)")
    
    print(f"Running {num_queries} search queries...")
    scalar_time = benchmark_search(num_queries, k=10)
    print(f"Scalar time: {scalar_time:.3f} seconds")
    print(f"Average per query: {scalar_time/num_queries*1000:.2f}ms")
    
    # Compare results
    print("\n" + "="*70)
    print("📈 PERFORMANCE COMPARISON")
    print("="*70)
    print(f"SIMD time:   {simd_time:.3f}s")
    print(f"Scalar time: {scalar_time:.3f}s")
    
    if simd_time < scalar_time:
        speedup = scalar_time / simd_time
        improvement = ((scalar_time - simd_time) / scalar_time) * 100
        print(f"\n🚀 SIMD is {speedup:.2f}x FASTER!")
        print(f"   Performance improvement: {improvement:.1f}%")
    elif simd_time > scalar_time:
        slowdown = simd_time / scalar_time
        print(f"\n⚠️  SIMD is {slowdown:.2f}x slower (unexpected - check implementation)")
    else:
        print(f"\n🤔 Same performance (might be due to network overhead)")
    
    # Re-enable SIMD for future use
    print("\n" + "="*70)
    print("🔄 RESTORING SIMD STATE")
    print("="*70)
    result = toggle_simd(True)
    print(f"✅ {result['message']}")
    
    # Final status
    print("\n" + "="*70)
    print("📊 FINAL SIMD STATUS")
    print("="*70)
    final_status = check_simd_status()
    print(f"SIMD Enabled: {final_status['simd_enabled']}")
    print(f"SIMD Type: {final_status['simd_type']}")
    
    print("\n" + "="*70)
    print("✅ SIMD TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # API Usage Examples
    print("\n" + "="*70)
    print("📚 API USAGE EXAMPLES")
    print("="*70)
    print("\n# Check SIMD status:")
    print(f"curl {BASE_URL}/config/simd")
    print("\n# Enable SIMD:")
    print(f"curl -X PUT {BASE_URL}/config/simd -H 'Content-Type: application/json' -d '{{\"enabled\": true}}'")
    print("\n# Disable SIMD (use scalar fallback):")
    print(f"curl -X PUT {BASE_URL}/config/simd -H 'Content-Type: application/json' -d '{{\"enabled\": false}}'")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
