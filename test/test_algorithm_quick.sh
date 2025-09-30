#!/bin/bash
# Quick algorithm switching test

BASE_URL="http://localhost:8080"

echo "========================================="
echo "Quick Algorithm Switching Test"
echo "========================================="

echo -e "\n1. Get current algorithm info:"
curl -s $BASE_URL/config/algorithm | jq -r '.current_algorithm'

echo -e "\n2. Switch to LSH:"
curl -s -X PUT $BASE_URL/config/algorithm \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "lsh", "num_tables": 10, "num_hash_functions": 8}' | jq

echo -e "\n3. Test search with LSH:"
curl -s -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "k": 5}' | jq

echo -e "\n4. Switch to HNSW (WARNING: May take 1-2 minutes to rebuild with 50K vectors):"
echo "   You can skip this and just use LSH which is already fast!"
read -p "   Press Enter to switch to HNSW (or Ctrl+C to skip)..."

curl -s -X PUT $BASE_URL/config/algorithm \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "hnsw", "M": 16, "ef_construction": 200}' | jq

echo -e "\n5. Test search with HNSW:"
curl -s -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "k": 5}' | jq

echo -e "\n6. Check statistics:"
curl -s $BASE_URL/statistics | jq '.database | {algorithm, total_vectors, total_searches}'

echo -e "\nDone!"
