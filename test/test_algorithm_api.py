#!/usr/bin/env python3
"""
Test script for Vector Database Algorithm Switching API

Tests:
1. Get current algorithm and available options
2. Switch to LSH with custom parameters
3. Switch to HNSW with custom parameters
4. Switch back to exact
5. Verify search performance differences
6. Test error handling (invalid algorithms)
"""

import requests
import json
import time
import random
import sys

BASE_URL = "http://localhost:8080"

def print_test(name):
    """Print test name"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)

def print_pass(message):
    """Print success message"""
    print(f"[PASS] PASS: {message}")

def print_fail(message):
    """Print failure message"""
    print(f"[FAIL] FAIL: {message}")
    
def print_info(message):
    """Print info message"""
    print(f"ℹ️  INFO: {message}")

def generate_random_vector(dims=128):
    """Generate random vector"""
    return [random.random() for _ in range(dims)]

def test_get_algorithm_info():
    """Test getting algorithm information"""
    print_test("Get Algorithm Information")
    
    response = requests.get(f"{BASE_URL}/config/algorithm")
    
    if response.status_code != 200:
        print_fail(f"Status code: {response.status_code}")
        return False
    
    data = response.json()
    print_info(f"Current algorithm: {data['current_algorithm']}")
    print_info(f"Available algorithms: {len(data['available_algorithms'])}")
    
    # Verify structure
    if 'current_algorithm' not in data:
        print_fail("Missing 'current_algorithm'")
        return False
    
    if 'available_algorithms' not in data:
        print_fail("Missing 'available_algorithms'")
        return False
    
    # Check each algorithm
    algorithms = data['available_algorithms']
    expected_algorithms = ['exact', 'lsh', 'hnsw']
    
    for algo in algorithms:
        name = algo['name']
        print_info(f"\n  {name.upper()}:")
        print_info(f"    - Description: {algo['description']}")
        print_info(f"    - Accuracy: {algo['accuracy']}")
        print_info(f"    - Speed: {algo['speed']}")
        print_info(f"    - Memory: {algo['memory']}")
        print_info(f"    - Use case: {algo['use_case']}")
        
        if name not in expected_algorithms:
            print_fail(f"Unexpected algorithm: {name}")
            return False
    
    print_pass("Algorithm information retrieved successfully")
    return True

def test_switch_to_lsh():
    """Test switching to LSH algorithm"""
    print_test("Switch to LSH Algorithm")
    
    request_data = {
        "algorithm": "lsh",
        "num_tables": 15,
        "num_hash_functions": 10
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 200:
        print_fail(f"Status code: {response.status_code}")
        print_info(f"Response: {response.text}")
        return False
    
    data = response.json()
    print_info(f"Response: {json.dumps(data, indent=2)}")
    
    if data['algorithm'] != 'lsh':
        print_fail(f"Expected 'lsh', got '{data['algorithm']}'")
        return False
    
    if data['parameters']['num_tables'] != 15:
        print_fail(f"Expected 15 tables, got {data['parameters']['num_tables']}")
        return False
    
    if data['parameters']['num_hash_functions'] != 10:
        print_fail(f"Expected 10 hash functions, got {data['parameters']['num_hash_functions']}")
        return False
    
    print_pass(f"Switched to LSH: {data['expected_performance']}")
    return True

def test_switch_to_hnsw():
    """Test switching to HNSW algorithm"""
    print_test("Switch to HNSW Algorithm")
    
    request_data = {
        "algorithm": "hnsw",
        "M": 20,
        "ef_construction": 250
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 200:
        print_fail(f"Status code: {response.status_code}")
        print_info(f"Response: {response.text}")
        return False
    
    data = response.json()
    print_info(f"Response: {json.dumps(data, indent=2)}")
    
    if data['algorithm'] != 'hnsw':
        print_fail(f"Expected 'hnsw', got '{data['algorithm']}'")
        return False
    
    if data['parameters']['M'] != 20:
        print_fail(f"Expected M=20, got {data['parameters']['M']}")
        return False
    
    if data['parameters']['ef_construction'] != 250:
        print_fail(f"Expected ef_construction=250, got {data['parameters']['ef_construction']}")
        return False
    
    print_pass(f"Switched to HNSW: {data['expected_performance']}")
    return True

def test_switch_to_exact():
    """Test switching back to exact search"""
    print_test("Switch to Exact Search")
    
    request_data = {
        "algorithm": "exact"
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 200:
        print_fail(f"Status code: {response.status_code}")
        print_info(f"Response: {response.text}")
        return False
    
    data = response.json()
    print_info(f"Response: {json.dumps(data, indent=2)}")
    
    if data['algorithm'] != 'exact':
        print_fail(f"Expected 'exact', got '{data['algorithm']}'")
        return False
    
    print_pass(f"Switched to exact: {data['expected_performance']}")
    return True

def test_performance_comparison():
    """Test performance difference between algorithms"""
    print_test("Performance Comparison Across Algorithms")
    
    # Insert test vectors using BATCH INSERT (much faster!)
    num_vectors = 5000
    batch_size = 1000
    print_info(f"Inserting {num_vectors} test vectors using batch insert (batch_size={batch_size})...")
    
    start_insert = time.time()
    
    for batch_start in range(0, num_vectors, batch_size):
        batch_end = min(batch_start + batch_size, num_vectors)
        batch_count = batch_end - batch_start
        
        # Generate batch data
        keys = [f"perf_test_{i}" for i in range(batch_start, batch_end)]
        vectors = [generate_random_vector(128) for _ in range(batch_count)]
        metadata = [f"Performance test vector {i}" for i in range(batch_start, batch_end)]
        
        # Batch insert
        response = requests.post(
            f"{BASE_URL}/vectors/batch/insert",
            json={
                "keys": keys,
                "vectors": vectors,
                "metadata": metadata
            }
        )
        
        if response.status_code != 200:
            print_fail(f"Batch insert failed: {response.text}")
            return False
        
        print_info(f"  Inserted {batch_end}/{num_vectors} vectors")
    
    insert_time = time.time() - start_insert
    print_info(f"Vectors inserted successfully in {insert_time:.2f}s ({num_vectors/insert_time:.0f} vectors/sec)")
    
    query_vector = generate_random_vector(128)
    num_searches = 100
    
    algorithms = [
        ("exact", {}),
        ("lsh", {"num_tables": 10, "num_hash_functions": 8}),
        ("hnsw", {"M": 16, "ef_construction": 200})
    ]
    
    results = {}
    
    for algo_name, params in algorithms:
        print_info(f"\nTesting {algo_name.upper()}...")
        
        # Switch algorithm
        request_data = {"algorithm": algo_name, **params}
        response = requests.put(f"{BASE_URL}/config/algorithm", json=request_data)
        
        if response.status_code != 200:
            print_fail(f"Failed to switch to {algo_name}")
            continue
        
        # Warm up
        for _ in range(5):
            requests.post(f"{BASE_URL}/search", json={"vector": query_vector, "k": 10})
        
        # Time searches
        start = time.time()
        for _ in range(num_searches):
            requests.post(f"{BASE_URL}/search", json={"vector": query_vector, "k": 10})
        elapsed = time.time() - start
        
        avg_latency = (elapsed / num_searches) * 1000  # ms
        results[algo_name] = avg_latency
        
        print_info(f"  Average latency: {avg_latency:.2f}ms")
    
    # Compare results
    print_info("\n📊 Performance Summary:")
    baseline = results.get('exact', 1.0)
    
    for algo_name, latency in sorted(results.items(), key=lambda x: x[1]):
        speedup = baseline / latency if latency > 0 else 1.0
        print_info(f"  {algo_name.upper()}: {latency:.2f}ms (speedup: {speedup:.2f}x)")
    
    print_pass("Performance comparison completed")
    return True

def test_invalid_algorithm():
    """Test error handling for invalid algorithm"""
    print_test("Invalid Algorithm Error Handling")
    
    request_data = {
        "algorithm": "invalid_algo"
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 400:
        print_fail(f"Expected 400, got {response.status_code}")
        return False
    
    data = response.json()
    print_info(f"Error response: {data['error']}")
    
    if 'error' not in data:
        print_fail("Missing error message")
        return False
    
    print_pass("Invalid algorithm rejected correctly")
    return True

def test_missing_algorithm_field():
    """Test error handling for missing algorithm field"""
    print_test("Missing Algorithm Field Error Handling")
    
    request_data = {
        "num_tables": 10  # Missing 'algorithm' field
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 400:
        print_fail(f"Expected 400, got {response.status_code}")
        return False
    
    data = response.json()
    print_info(f"Error response: {data['error']}")
    
    if 'error' not in data:
        print_fail("Missing error message")
        return False
    
    print_pass("Missing field rejected correctly")
    return True

def test_default_parameters():
    """Test switching with default parameters"""
    print_test("Default Parameters")
    
    # Switch to HNSW without specifying parameters
    request_data = {
        "algorithm": "hnsw"
    }
    
    response = requests.put(
        f"{BASE_URL}/config/algorithm",
        json=request_data
    )
    
    if response.status_code != 200:
        print_fail(f"Status code: {response.status_code}")
        return False
    
    data = response.json()
    print_info(f"Default parameters: {data['parameters']}")
    
    # Check defaults are applied
    if 'M' not in data['parameters']:
        print_fail("Missing M parameter")
        return False
    
    if 'ef_construction' not in data['parameters']:
        print_fail("Missing ef_construction parameter")
        return False
    
    print_pass(f"Default parameters applied: M={data['parameters']['M']}, ef_construction={data['parameters']['ef_construction']}")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("VECTOR DATABASE - ALGORITHM SWITCHING API TESTS")
    print("="*60)
    
    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print_fail("Server is not healthy")
            return
    except requests.exceptions.ConnectionError:
        print_fail(f"Cannot connect to server at {BASE_URL}")
        print_info("Make sure the server is running:")
        print_info("  cd /Users/habibrahman/Code/vector_database/build")
        print_info("  ./vector_db_server")
        return
    
    tests = [
        test_get_algorithm_info,
        test_switch_to_lsh,
        test_switch_to_hnsw,
        test_switch_to_exact,
        test_default_parameters,
        test_invalid_algorithm,
        test_missing_algorithm_field,
        test_performance_comparison,  # Last because it inserts many vectors
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_fail(f"Exception: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"[PASS] Passed: {passed}/{len(tests)}")
    print(f"[FAIL] Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n  {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
