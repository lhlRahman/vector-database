#!/bin/bash

# Usage: ./insert_vectors.sh <number_of_vectors>
# Example: ./insert_vectors.sh 10

if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_vectors>"
  exit 1
fi

N=$1
VECTOR_SIZE=128
URL="http://localhost:8080/vectors"

for ((i=1; i<=N; i++)); do
  # Generate a random 128-dim vector
  VECTOR=$(awk -v size=$VECTOR_SIZE 'BEGIN {
    srand();
    printf("[");
    for (j=1; j<=size; j++) {
      val = rand();   # random float between 0 and 1
      printf("%f", val);
      if (j < size) printf(", ");
    }
    printf("]");
  }')

  # Create JSON payload
  JSON=$(jq -n --arg key "vec_$i" --argjson vector "$VECTOR" \
    '{key: $key, vector: $vector}')

  # Insert with curl
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "$JSON"

  echo " -> Inserted vector $i"
done
