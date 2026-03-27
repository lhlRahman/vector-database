#!/usr/bin/env python3
"""
API test for update/delete correctness.
Ensures updated vectors are searchable and deleted vectors are not returned.
"""

import requests
import sys

BASE_URL = "http://localhost:8080"
DIMENSIONS = 128


def check_server():
    try:
        return requests.get(f"{BASE_URL}/health").status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def set_exact_algorithm():
    res = requests.put(f"{BASE_URL}/config/algorithm", json={"algorithm": "exact"})
    res.raise_for_status()


def insert_vector(key, vector):
    res = requests.post(f"{BASE_URL}/vectors", json={"key": key, "vector": vector})
    return res.status_code == 200


def update_vector(key, vector):
    res = requests.put(f"{BASE_URL}/vectors/{key}", json={"vector": vector})
    return res.status_code == 200


def delete_vector(key):
    res = requests.delete(f"{BASE_URL}/vectors/{key}")
    return res.status_code == 200


def search(query, k=3):
    res = requests.post(f"{BASE_URL}/search", json={"query": query, "k": k})
    res.raise_for_status()
    return res.json()


def main():
    if not check_server():
        print("[FAIL] Server is not running. Start it with: ./build/vector_db_server")
        return 1

    set_exact_algorithm()

    key = "update_delete_test_vec"
    vec_a = [0.1] * DIMENSIONS
    vec_b = [0.9] * DIMENSIONS

    if not insert_vector(key, vec_a):
        print("[FAIL] Insert failed")
        return 1

    results = search(vec_a, k=1)["results"]
    if not results or results[0]["key"] != key:
        print("[FAIL] Inserted vector not returned in search")
        return 1

    if not update_vector(key, vec_b):
        print("[FAIL] Update failed")
        return 1

    results = search(vec_b, k=1)["results"]
    if not results or results[0]["key"] != key:
        print("[FAIL] Updated vector not returned in search")
        return 1

    if not delete_vector(key):
        print("[FAIL] Delete failed")
        return 1

    results = search(vec_b, k=3)["results"]
    if any(r["key"] == key for r in results):
        print("[FAIL] Deleted vector still returned in search")
        return 1

    print("[PASS] Update/Delete correctness test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
