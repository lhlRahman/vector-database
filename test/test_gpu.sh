#!/bin/bash

# Test GPU vs CPU BATCH search performance
# Usage: ./test_gpu.sh [dimensions] [k] [batch_size]

HOST="localhost"
PORT="8080"
BASE_URL="http://${HOST}:${PORT}"
DIMS=${1:-128}
K=${2:-10}
BATCH_SIZE=${3:-50}

echo "================================================"
echo "GPU vs CPU BATCH Search Performance Test"
echo "================================================"
echo "Dimensions: $DIMS"
echo "K (neighbors): $K"
echo "Batch size: $BATCH_SIZE queries"
echo "================================================"

# Check current status
echo ""
echo "1. Current Database Status:"
curl -s "${BASE_URL}/statistics/database" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'   Total vectors: {d.get(\"total_vectors\", \"N/A\")}')
    print(f'   Algorithm: {d.get(\"algorithm\", \"N/A\")}')
    print(f'   Dimensions: {d.get(\"dimensions\", \"N/A\")}')
except: print('   Could not parse stats')
"

echo ""
echo "2. Current GPU Status:"
curl -s "${BASE_URL}/config/gpu" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'   GPU available: {d.get(\"gpu_available\", \"N/A\")}')
    print(f'   GPU enabled: {d.get(\"gpu_enabled\", \"N/A\")}')
    print(f'   GPU threshold: {d.get(\"gpu_threshold\", \"N/A\")}')
except: print('   Could not parse GPU status')
"

# Generate batch of random query vectors using Python (much faster)
echo ""
echo "Generating $BATCH_SIZE random query vectors..."

BATCH_QUERIES=$(python3 -c "
import random
import json

dims = $DIMS
batch_size = $BATCH_SIZE

queries = []
for _ in range(batch_size):
    vec = [round(random.uniform(-1, 1), 6) for _ in range(dims)]
    queries.append(vec)

print(json.dumps(queries))
")

echo "Done generating queries."

echo ""
echo "================================================"
echo "3. Testing BATCH search with GPU DISABLED (CPU)"
echo "================================================"

# Disable GPU
curl -s -X PUT "${BASE_URL}/config/gpu" \
    -H "Content-Type: application/json" \
    -d '{"enabled": false}' > /dev/null

echo "GPU disabled. Running batch search ($BATCH_SIZE queries)..."

# Run batch search 3 times and average
cpu_times=()
# Write queries to temp file to avoid argument list too long
QUERY_FILE=$(mktemp)
echo "{\"queries\": ${BATCH_QUERIES}, \"k\": ${K}}" > "$QUERY_FILE"

# Warmup run (not counted)
echo "  Warmup run..."
curl -s -X POST "${BASE_URL}/search/batch" \
    -H "Content-Type: application/json" \
    -d @"$QUERY_FILE" > /dev/null

for run in {1..3}; do
    start=$(python3 -c "import time; print(time.time())")
    
    result=$(curl -s -X POST "${BASE_URL}/search/batch" \
        -H "Content-Type: application/json" \
        -d @"$QUERY_FILE")
    
    end=$(python3 -c "import time; print(time.time())")
    duration=$(python3 -c "print(f'{($end - $start) * 1000:.2f}')")
    
    echo "  Run $run: ${duration}ms (${BATCH_SIZE} queries)"
    cpu_times+=($duration)
done

cpu_avg=$(python3 -c "times=[${cpu_times[0]}, ${cpu_times[1]}, ${cpu_times[2]}]; print(f'{sum(times)/len(times):.2f}')")
cpu_per_query=$(python3 -c "print(f'{$cpu_avg / $BATCH_SIZE:.2f}')")
echo ""
echo "CPU Total: ${cpu_avg}ms for $BATCH_SIZE queries"
echo "CPU Per Query: ${cpu_per_query}ms"

echo ""
echo "================================================"
echo "4. Testing BATCH search with GPU ENABLED"
echo "================================================"

# Enable GPU
curl -s -X PUT "${BASE_URL}/config/gpu" \
    -H "Content-Type: application/json" \
    -d '{"enabled": true, "threshold": 1000}' > /dev/null

echo "GPU enabled (threshold: 1000). Running batch search ($BATCH_SIZE queries)..."

gpu_times=()

# Warmup run (not counted) - also triggers GPU buffer rebuild
echo "  Warmup run (+ GPU buffer rebuild)..."
curl -s -X POST "${BASE_URL}/search/batch" \
    -H "Content-Type: application/json" \
    -d @"$QUERY_FILE" > /dev/null

for run in {1..3}; do
    start=$(python3 -c "import time; print(time.time())")
    
    result=$(curl -s -X POST "${BASE_URL}/search/batch" \
        -H "Content-Type: application/json" \
        -d @"$QUERY_FILE")
    
    end=$(python3 -c "import time; print(time.time())")
    duration=$(python3 -c "print(f'{($end - $start) * 1000:.2f}')")
    
    echo "  Run $run: ${duration}ms (${BATCH_SIZE} queries)"
    gpu_times+=($duration)
done

# Cleanup temp file
rm -f "$QUERY_FILE"

gpu_avg=$(python3 -c "times=[${gpu_times[0]}, ${gpu_times[1]}, ${gpu_times[2]}]; print(f'{sum(times)/len(times):.2f}')")
gpu_per_query=$(python3 -c "print(f'{$gpu_avg / $BATCH_SIZE:.2f}')")
echo ""
echo "GPU Total: ${gpu_avg}ms for $BATCH_SIZE queries"
echo "GPU Per Query: ${gpu_per_query}ms"

echo ""
echo "================================================"
echo "5. Results Summary"
echo "================================================"
echo ""
echo "                    CPU (HNSW)    GPU (Metal)"
echo "  ─────────────────────────────────────────────"
printf "  Total time:       %10sms  %10sms\n" "$cpu_avg" "$gpu_avg"
printf "  Per query:        %10sms  %10sms\n" "$cpu_per_query" "$gpu_per_query"
echo "  ─────────────────────────────────────────────"

speedup=$(python3 -c "print(f'{$cpu_avg / $gpu_avg:.2f}')" 2>/dev/null || echo "N/A")
echo ""
if (( $(echo "$cpu_avg > $gpu_avg" | bc -l) )); then
    echo "  GPU is ${speedup}x FASTER"
else
    slowdown=$(python3 -c "print(f'{$gpu_avg / $cpu_avg:.2f}')" 2>/dev/null || echo "N/A")
    echo "  GPU is ${slowdown}x slower (overhead > parallelism benefit)"
    echo ""
    echo "  This can happen because:"
    echo "    - GPU brute-force checks ALL vectors"
    echo "    - HNSW only checks ~100-500 vectors (approximate)"
    echo "    - Data copy overhead to GPU"
    echo ""
    echo "  GPU wins when:"
    echo "    - Dataset is 50K-100K+ vectors"
    echo "    - You need exact (100%) results"
    echo "    - Doing many batch queries"
fi

echo ""
echo "================================================"
