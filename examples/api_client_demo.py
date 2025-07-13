#!/usr/bin/env python3
"""
Vector Database API Client Demo

This script demonstrates how to interact with the Vector Database REST API using Python.

Requirements:
    pip install requests numpy
"""

import requests
import numpy as np
import json
from typing import List, Dict, Any

class VectorDBClient:
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """Get database information"""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def insert_vector(self, key: str, vector: List[float], metadata: str = None) -> Dict[str, Any]:
        """Insert a single vector"""
        data = {
            "key": key,
            "vector": vector
        }
        if metadata:
            data["metadata"] = metadata
        
        response = self.session.post(f"{self.base_url}/vectors", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_insert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch insert multiple vectors"""
        data = {"vectors": vectors}
        response = self.session.post(f"{self.base_url}/vectors/batch", json=data)
        response.raise_for_status()
        return response.json()
    
    def search(self, vector: List[float], k: int = 5, with_metadata: bool = False) -> Dict[str, Any]:
        """Search for similar vectors"""
        data = {
            "vector": vector,
            "k": k,
            "with_metadata": with_metadata
        }
        response = self.session.post(f"{self.base_url}/search", json=data)
        response.raise_for_status()
        return response.json()
    
    def toggle_approximate_search(self, enabled: bool) -> Dict[str, Any]:
        """Toggle between exact and approximate search"""
        data = {"enabled": enabled}
        response = self.session.put(f"{self.base_url}/config/approximate", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_vector(self, key: str) -> Dict[str, Any]:
        """Get a specific vector by key"""
        response = self.session.get(f"{self.base_url}/vectors/{key}")
        response.raise_for_status()
        return response.json()
    
    def list_vectors(self, page: int = 1, per_page: int = 100) -> Dict[str, Any]:
        """List all vectors with pagination"""
        params = {"page": page, "per_page": per_page}
        response = self.session.get(f"{self.base_url}/vectors", params=params)
        response.raise_for_status()
        return response.json()
    
    def save_database(self) -> Dict[str, Any]:
        """Save database to disk"""
        response = self.session.post(f"{self.base_url}/save")
        response.raise_for_status()
        return response.json()


def generate_random_vector(dimensions: int) -> List[float]:
    """Generate a random normalized vector"""
    vec = np.random.randn(dimensions)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def main():
    print("=== Vector Database API Python Client Demo ===\n")
    
    # Initialize client
    client = VectorDBClient()
    
    # 1. Check server health
    print("1. Checking server health...")
    try:
        health = client.health_check()
        print(f"   Server status: {health['status']}")
        print(f"   Service: {health['service']}")
        print(f"   Version: {health['version']}")
    except Exception as e:
        print(f"   Failed to connect to server: {e}")
        return
    
    # 2. Get database info
    print("\n2. Getting database info...")
    info = client.get_info()
    print(f"   Dimensions: {info['dimensions']}")
    print(f"   Approximate search: {info['use_approximate']}")
    print(f"   Vector count: {info['vector_count']}")
    
    # 3. Insert single vector
    print("\n3. Inserting a single vector...")
    vector = generate_random_vector(128)
    result = client.insert_vector("python_vector_1", vector, "Created from Python client")
    print(f"   Status: {result['status']}")
    print(f"   Key: {result['key']}")
    
    # 4. Batch insert vectors
    print("\n4. Batch inserting 10 vectors...")
    batch_data = []
    for i in range(2, 12):
        batch_data.append({
            "key": f"python_vector_{i}",
            "vector": generate_random_vector(128)
        })
    
    result = client.batch_insert(batch_data)
    print(f"   Status: {result['status']}")
    print(f"   Count: {result['count']}")
    
    # 5. Search for similar vectors (exact)
    print("\n5. Searching for similar vectors (exact search)...")
    query_vector = generate_random_vector(128)
    results = client.search(query_vector, k=5, with_metadata=True)
    print(f"   Found {results['count']} similar vectors:")
    
    for res in results['results'][:3]:  # Show top 3
        print(f"   - Key: {res['key']}, Distance: {res['distance']:.4f}", end="")
        if 'metadata' in res and res['metadata']:
            print(f", Metadata: {res['metadata']}", end="")
        print()
    
    # 6. Toggle to approximate search
    print("\n6. Enabling approximate search...")
    result = client.toggle_approximate_search(True)
    print(f"   Status: {result['status']}")
    print(f"   Approximate search: {result['approximate_search']}")
    
    # 7. Search again with approximate search
    print("\n7. Searching with approximate search (LSH)...")
    results = client.search(query_vector, k=5)
    print(f"   Found {results['count']} similar vectors:")
    
    for res in results['results'][:3]:  # Show top 3
        print(f"   - Key: {res['key']}, Distance: {res['distance']:.4f}")
    
    # 8. Get specific vector
    print("\n8. Getting specific vector...")
    vec_data = client.get_vector("python_vector_1")
    print(f"   Key: {vec_data['key']}")
    print(f"   Dimensions: {len(vec_data['vector'])}")
    if 'metadata' in vec_data:
        print(f"   Metadata: {vec_data['metadata']}")
    
    # 9. List vectors with pagination
    print("\n9. Listing vectors (page 1)...")
    listing = client.list_vectors(page=1, per_page=5)
    print(f"   Total vectors: {listing['total']}")
    print(f"   Page: {listing['page']}/{listing['total_pages']}")
    print("   Vectors on this page:")
    for vec in listing['vectors']:
        print(f"   - {vec['key']}")
    
    # 10. Save database
    print("\n10. Saving database to disk...")
    result = client.save_database()
    print(f"   Status: {result['status']}")
    print(f"   File: {result['file']}")
    
    # Additional examples
    print("\n=== Additional Examples ===")
    
    # Example: Text embedding simulation
    print("\n11. Simulating text embeddings...")
    text_embeddings = {
        "doc_hello": "Hello world document",
        "doc_python": "Python programming language",
        "doc_vector": "Vector database tutorial"
    }
    
    for key, text in text_embeddings.items():
        # In practice, you would use a real embedding model here
        fake_embedding = generate_random_vector(128)
        client.insert_vector(key, fake_embedding, text)
        print(f"   Inserted: {key} - {text}")
    
    # Search for similar documents
    print("\n12. Finding similar documents...")
    query_embedding = generate_random_vector(128)
    results = client.search(query_embedding, k=3, with_metadata=True)
    print("   Similar documents:")
    for res in results['results']:
        if res['key'].startswith('doc_'):
            print(f"   - {res['key']}: {res.get('metadata', 'No metadata')}")
            print(f"     Distance: {res['distance']:.4f}")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()