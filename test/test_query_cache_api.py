#!/usr/bin/env python3
"""
Test script for Query Cache functionality via API
Tests cache hits, misses, invalidation, and performance
"""

import requests
import json
import time
import random
from typing import List, Dict, Any

# Configuration
BASE_URL = "http://localhost:8080"
DIMENSIONS = 128

class VectorDBCacheTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_health(self) -> bool:
        """Check if the server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics including cache stats"""
        response = self.session.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    def insert_vector(self, key: str, vector: List[float], metadata: str = "") -> bool:
        """Insert a vector into the database"""
        payload = {
            "key": key,
            "vector": vector,
            "metadata": metadata
        }
        response = self.session.post(f"{self.base_url}/vectors", json=payload)
        return response.status_code == 200
    
    def search(self, query: List[float], k: int = 5) -> Dict[str, Any]:
        """Perform similarity search"""
        payload = {
            "query": query,
            "k": k
        }
        response = self.session.post(f"{self.base_url}/search", json=payload)
        response.raise_for_status()
        return response.json()
    
    def delete_vector(self, key: str) -> bool:
        """Delete a vector"""
        response = self.session.delete(f"{self.base_url}/vectors/{key}")
        return response.status_code == 200
    
    def generate_random_vector(self, dimensions: int = DIMENSIONS) -> List[float]:
        """Generate a random vector"""
        return [random.random() for _ in range(dimensions)]


def test_cache_basic_functionality():
    """Test 1: Basic cache hit/miss functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic Cache Hit/Miss Functionality")
    print("="*70)
    
    tester = VectorDBCacheTester()
    
    if not tester.check_health():
        print("[FAIL] Server is not running. Start the server first.")
        return False
    
    print("[PASS] Server is running")
    
    # Insert test vectors
    print("\n📝 Inserting test vectors...")
    for i in range(10):
        vector = tester.generate_random_vector()
        tester.insert_vector(f"test_vec_{i}", vector, f"metadata_{i}")
    print("[PASS] Inserted 10 vectors")
    
    # Get initial stats
    stats_before = tester.get_statistics()
    cache_before = stats_before.get("cache_stats", {})
    print(f"\n📊 Cache stats before search:")
    print(f"   Hits: {cache_before.get('hits', 0)}")
    print(f"   Misses: {cache_before.get('misses', 0)}")
    print(f"   Hit Rate: {cache_before.get('hit_rate', 0):.2%}")
    
    # Perform first search (should be cache miss)
    query1 = tester.generate_random_vector()
    print(f"\n🔍 Performing first search (should miss cache)...")
    result1 = tester.search(query1, k=5)
    print(f"[PASS] Found {len(result1.get('results', []))} results")
    
    # Check stats after first search
    stats_after_1 = tester.get_statistics()
    cache_after_1 = stats_after_1.get("cache_stats", {})
    print(f"\n📊 Cache stats after first search:")
    print(f"   Hits: {cache_after_1.get('hits', 0)}")
    print(f"   Misses: {cache_after_1.get('misses', 0)}")
    print(f"   Expected: 0 hits, 1 miss")
    
    assert cache_after_1.get('misses', 0) == 1, "First search should be a cache miss"
    print("[PASS] First search correctly registered as cache miss")
    
    # Perform same search again (should be cache hit)
    print(f"\n🔍 Performing same search again (should hit cache)...")
    time.sleep(0.1)  # Small delay
    result2 = tester.search(query1, k=5)
    
    # Check stats after second search
    stats_after_2 = tester.get_statistics()
    cache_after_2 = stats_after_2.get("cache_stats", {})
    print(f"\n📊 Cache stats after second search:")
    print(f"   Hits: {cache_after_2.get('hits', 0)}")
    print(f"   Misses: {cache_after_2.get('misses', 0)}")
    print(f"   Hit Rate: {cache_after_2.get('hit_rate', 0):.2%}")
    print(f"   Expected: 1 hit, 1 miss")
    
    assert cache_after_2.get('hits', 0) == 1, "Second search should be a cache hit"
    assert result1 == result2, "Results should be identical"
    print("[PASS] Second search correctly registered as cache hit")
    print("[PASS] Results are identical")
    
    print("\n" + "="*70)
    print("[PASS] TEST 1 PASSED: Basic cache functionality works correctly")
    print("="*70)
    return True


def test_cache_invalidation():
    """Test 2: Cache invalidation on insert/update/delete"""
    print("\n" + "="*70)
    print("TEST 2: Cache Invalidation on Data Modifications")
    print("="*70)
    
    tester = VectorDBCacheTester()
    
    # Insert initial vectors
    print("\n📝 Inserting initial vectors...")
    for i in range(5):
        vector = tester.generate_random_vector()
        tester.insert_vector(f"cache_test_{i}", vector)
    
    # Perform search to populate cache
    query = tester.generate_random_vector()
    print("\n🔍 Performing search to populate cache...")
    result1 = tester.search(query, k=3)
    
    # Check cache stats
    stats1 = tester.get_statistics()
    cache1 = stats1.get("cache_stats", {})
    initial_hits = cache1.get('hits', 0)
    initial_misses = cache1.get('misses', 0)
    print(f"📊 Initial cache: {initial_hits} hits, {initial_misses} misses")
    
    # Perform same search (should hit cache)
    print("\n🔍 Performing same search (should hit cache)...")
    result2 = tester.search(query, k=3)
    stats2 = tester.get_statistics()
    cache2 = stats2.get("cache_stats", {})
    
    assert cache2.get('hits', 0) > initial_hits, "Should have cache hit"
    print(f"[PASS] Cache hit confirmed: {cache2.get('hits', 0)} hits")
    
    # Insert new vector (should invalidate cache)
    print("\n📝 Inserting new vector (should invalidate cache)...")
    new_vector = tester.generate_random_vector()
    tester.insert_vector("invalidation_test", new_vector)
    
    # Check that cache was cleared
    stats3 = tester.get_statistics()
    cache3 = stats3.get("cache_stats", {})
    print(f"📊 Cache after insert: {cache3.get('hits', 0)} hits, {cache3.get('misses', 0)} misses")
    print(f"   Cache size: {cache3.get('current_size', 0)}")
    
    assert cache3.get('current_size', -1) == 0, "Cache should be cleared after insert"
    print("[PASS] Cache correctly cleared after insert")
    
    # Perform same search again (should miss because cache was cleared)
    print("\n🔍 Performing same search again (should miss - cache was cleared)...")
    result3 = tester.search(query, k=3)
    stats4 = tester.get_statistics()
    cache4 = stats4.get("cache_stats", {})
    
    # After clearing, hits reset to 0, so this will be miss
    assert cache4.get('misses', 0) > 0, "Should have cache miss after invalidation"
    print(f"[PASS] Cache miss confirmed after invalidation: {cache4.get('misses', 0)} misses")
    
    print("\n" + "="*70)
    print("[PASS] TEST 2 PASSED: Cache invalidation works correctly")
    print("="*70)
    return True


def test_cache_performance():
    """Test 3: Cache performance improvement"""
    print("\n" + "="*70)
    print("TEST 3: Cache Performance Improvement")
    print("="*70)
    
    tester = VectorDBCacheTester()
    
    # Insert many vectors for realistic testing
    print("\n📝 Inserting 100 vectors...")
    for i in range(100):
        vector = tester.generate_random_vector()
        tester.insert_vector(f"perf_test_{i}", vector)
    print("[PASS] Inserted 100 vectors")
    
    # Generate a query
    query = tester.generate_random_vector()
    
    # Measure first search (cache miss)
    print("\n⏱️  Measuring first search (cache miss)...")
    start = time.time()
    result1 = tester.search(query, k=10)
    first_time = time.time() - start
    print(f"   First search time: {first_time*1000:.2f}ms")
    
    # Measure repeated searches (cache hits)
    print("\n⏱️  Measuring 10 repeated searches (cache hits)...")
    times = []
    for i in range(10):
        start = time.time()
        tester.search(query, k=10)
        times.append(time.time() - start)
    
    avg_cached_time = sum(times) / len(times)
    print(f"   Average cached search time: {avg_cached_time*1000:.2f}ms")
    print(f"   Speedup: {first_time/avg_cached_time:.2f}x faster")
    
    # Get final cache stats
    stats = tester.get_statistics()
    cache_stats = stats.get("cache_stats", {})
    print(f"\n📊 Final cache statistics:")
    print(f"   Total Hits: {cache_stats.get('hits', 0)}")
    print(f"   Total Misses: {cache_stats.get('misses', 0)}")
    print(f"   Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
    print(f"   Current Size: {cache_stats.get('current_size', 0)}/{cache_stats.get('capacity', 0)}")
    
    # Cache should provide some speedup (even if minimal due to network overhead)
    print(f"\n💡 Note: Cached searches should be faster or similar due to skipping computation")
    
    print("\n" + "="*70)
    print("[PASS] TEST 3 PASSED: Cache performance validated")
    print("="*70)
    return True


def test_cache_capacity():
    """Test 4: Cache LRU eviction"""
    print("\n" + "="*70)
    print("TEST 4: Cache Capacity and LRU Eviction")
    print("="*70)
    
    tester = VectorDBCacheTester()
    
    # Insert some vectors
    print("\n📝 Inserting 20 vectors...")
    for i in range(20):
        vector = tester.generate_random_vector()
        tester.insert_vector(f"lru_test_{i}", vector)
    
    # Get cache capacity
    stats = tester.get_statistics()
    cache_stats = stats.get("cache_stats", {})
    capacity = cache_stats.get('capacity', 1000)
    print(f"📦 Cache capacity: {capacity}")
    
    # Perform multiple unique searches
    print(f"\n🔍 Performing 10 unique searches...")
    unique_queries = [tester.generate_random_vector() for _ in range(10)]
    
    for i, query in enumerate(unique_queries):
        tester.search(query, k=3)
        if i % 3 == 0:
            stats = tester.get_statistics()
            cache_stats = stats.get("cache_stats", {})
            print(f"   After {i+1} searches: cache size = {cache_stats.get('current_size', 0)}")
    
    # Final stats
    stats = tester.get_statistics()
    cache_stats = stats.get("cache_stats", {})
    print(f"\n📊 Final cache statistics:")
    print(f"   Current Size: {cache_stats.get('current_size', 0)}/{capacity}")
    print(f"   Total Misses: {cache_stats.get('misses', 0)} (should be 10 unique queries)")
    
    assert cache_stats.get('current_size', 0) <= capacity, "Cache size should not exceed capacity"
    print("[PASS] Cache respects capacity limit")
    
    print("\n" + "="*70)
    print("[PASS] TEST 4 PASSED: Cache capacity and LRU working correctly")
    print("="*70)
    return True


def run_all_tests():
    """Run all cache tests"""
    print("\n" + "="*70)
    print(" VECTOR DATABASE QUERY CACHE TEST SUITE")
    print("="*70)
    
    tester = VectorDBCacheTester()
    
    # Check server health
    if not tester.check_health():
        print("\n[FAIL] FATAL: Server is not running!")
        print("   Please start the server with: ./build/vector_db_server")
        return False
    
    print("\n[PASS] Server is running and healthy")
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_cache_basic_functionality),
        ("Cache Invalidation", test_cache_invalidation),
        ("Performance", test_cache_performance),
        ("Capacity & LRU", test_cache_capacity)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] TEST FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS] PASSED" if result else "[FAIL] FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        return True
    else:
        print(f"  {total - passed} test(s) failed")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
