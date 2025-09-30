#!/usr/bin/env python3
"""
Test script for Distance Metrics API
Tests switching between Euclidean, Manhattan, and Cosine distance metrics
"""

import requests
import json
import random
from typing import List, Tuple

# Configuration
BASE_URL = "http://localhost:8080"
DIMENSIONS = 128

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_distance_metric():
    """Get current distance metric configuration"""
    response = requests.get(f"{BASE_URL}/config/distance-metric")
    response.raise_for_status()
    return response.json()

def set_distance_metric(metric: str):
    """Set distance metric"""
    response = requests.put(
        f"{BASE_URL}/config/distance-metric",
        json={"metric": metric}
    )
    response.raise_for_status()
    return response.json()

def insert_vector(key: str, vector: List[float], metadata: str = ""):
    """Insert a vector"""
    response = requests.post(
        f"{BASE_URL}/vectors",
        json={"key": key, "vector": vector, "metadata": metadata}
    )
    return response.status_code == 200

def search(query: List[float], k: int = 5):
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

def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length"""
    magnitude = sum(x**2 for x in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]

def main():
    print("\n" + "="*70)
    print("🎯 DISTANCE METRICS TEST SUITE")
    print("="*70)
    
    # Check server
    if not check_server():
        print("❌ Server is not running! Start it with: ./build/vector_db_server")
        return
    
    print("✅ Server is running\n")
    
    # Get initial configuration
    print("="*70)
    print("📊 INITIAL CONFIGURATION")
    print("="*70)
    config = get_distance_metric()
    print(f"Current metric: {config['current_metric']}")
    print(f"\nAvailable metrics:")
    for metric in config['available_metrics']:
        simd_status = "✅ SIMD" if metric['simd_accelerated'] else "⚠️  Scalar"
        print(f"  • {metric['name']:12} - {metric['description']}")
        print(f"    {simd_status:12}   Use case: {metric['use_case']}")
    
    # Insert test vectors
    print("\n" + "="*70)
    print("📝 INSERTING TEST VECTORS")
    print("="*70)
    
    # Create distinct test vectors
    test_vectors = {
        "v1": [1.0] * DIMENSIONS,  # All ones
        "v2": [2.0] * DIMENSIONS,  # All twos
        "v3": [0.5] * DIMENSIONS,  # All 0.5
        "v4": normalize_vector([random.random() for _ in range(DIMENSIONS)]),  # Random normalized
        "v5": normalize_vector([random.random() for _ in range(DIMENSIONS)]),  # Random normalized
    }
    
    for key, vector in test_vectors.items():
        insert_vector(key, vector, f"test vector {key}")
        print(f"  ✅ Inserted {key}")
    
    query = normalize_vector([1.5] * DIMENSIONS)  # Query vector
    
    # Test each distance metric
    metrics = ["euclidean", "manhattan", "cosine"]
    results = {}
    
    for metric in metrics:
        print("\n" + "="*70)
        print(f"🔍 TESTING: {metric.upper()} DISTANCE")
        print("="*70)
        
        # Switch metric
        response = set_distance_metric(metric)
        print(f"✅ {response['message']}")
        
        # Perform search
        search_results = search(query, k=5)
        results[metric] = search_results['results']
        
        print(f"\nTop {len(search_results['results'])} results:")
        for i, result in enumerate(search_results['results'], 1):
            print(f"  {i}. {result['key']:8} - distance: {result['distance']:.6f}")
    
    # Compare results
    print("\n" + "="*70)
    print("📈 COMPARISON OF DISTANCE METRICS")
    print("="*70)
    
    print("\nRankings by metric:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Euclidean':<15} {'Manhattan':<15} {'Cosine':<15}")
    print("-" * 70)
    
    max_rank = max(len(results[m]) for m in metrics)
    for i in range(max_rank):
        row = [str(i+1)]
        for metric in metrics:
            if i < len(results[metric]):
                key = results[metric][i]['key']
                dist = results[metric][i]['distance']
                row.append(f"{key} ({dist:.4f})")
            else:
                row.append("-")
        print(f"{row[0]:<6} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    # Analysis
    print("\n" + "="*70)
    print("🔬 ANALYSIS")
    print("="*70)
    
    # Check if rankings differ
    euclidean_order = [r['key'] for r in results['euclidean']]
    manhattan_order = [r['key'] for r in results['manhattan']]
    cosine_order = [r['key'] for r in results['cosine']]
    
    print(f"\n📊 Ranking Differences:")
    print(f"  Euclidean vs Manhattan: {'Different ✓' if euclidean_order != manhattan_order else 'Same'}")
    print(f"  Euclidean vs Cosine:    {'Different ✓' if euclidean_order != cosine_order else 'Same'}")
    print(f"  Manhattan vs Cosine:    {'Different ✓' if manhattan_order != cosine_order else 'Same'}")
    
    print(f"\n💡 Observations:")
    print(f"  • Cosine similarity is normalized - good for text embeddings")
    print(f"  • Manhattan distance treats each dimension independently")
    print(f"  • Euclidean distance is the most common general-purpose metric")
    
    # Test error handling
    print("\n" + "="*70)
    print("🧪 ERROR HANDLING TEST")
    print("="*70)
    
    try:
        response = requests.put(
            f"{BASE_URL}/config/distance-metric",
            json={"metric": "invalid_metric"}
        )
        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ Correctly rejected invalid metric:")
            print(f"   Error: {error_data['error']}")
            print(f"   Available: {error_data.get('available_metrics', [])}")
        else:
            print(f"❌ Expected 400 error, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid metric: {e}")
    
    # Restore to Euclidean
    print("\n" + "="*70)
    print("🔄 RESTORING DEFAULT METRIC")
    print("="*70)
    response = set_distance_metric("euclidean")
    print(f"✅ {response['message']}")
    
    # Final status
    print("\n" + "="*70)
    print("📊 FINAL STATUS")
    print("="*70)
    final_config = get_distance_metric()
    print(f"Current metric: {final_config['current_metric']}")
    
    print("\n" + "="*70)
    print("✅ ALL DISTANCE METRIC TESTS COMPLETED!")
    print("="*70)
    
    # API Usage Examples
    print("\n" + "="*70)
    print("📚 API USAGE EXAMPLES")
    print("="*70)
    
    print("\n# Check current distance metric:")
    print(f"curl {BASE_URL}/config/distance-metric")
    
    print("\n# Switch to Manhattan distance:")
    print(f"curl -X PUT {BASE_URL}/config/distance-metric \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"metric\": \"manhattan\"}}'")
    
    print("\n# Switch to Cosine similarity:")
    print(f"curl -X PUT {BASE_URL}/config/distance-metric \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"metric\": \"cosine\"}}'")
    
    print("\n# Switch back to Euclidean:")
    print(f"curl -X PUT {BASE_URL}/config/distance-metric \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"metric\": \"euclidean\"}}'")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
