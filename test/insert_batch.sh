#!/bin/bash

# Fast batch insert script
# Usage: ./insert_batch.sh <total_vectors> [dimensions] [batch_size]
# Example: ./insert_batch.sh 50000 128 1000

TOTAL=${1:-10000}
DIMS=${2:-128}
BATCH_SIZE=${3:-1000}
URL="http://localhost:8080/vectors/batch/insert"

echo "================================================"
echo "Fast Batch Insert"
echo "================================================"
echo "Total vectors: $TOTAL"
echo "Dimensions: $DIMS"
echo "Batch size: $BATCH_SIZE"
echo "================================================"

# Check server
if ! curl -s "http://localhost:8080/health" > /dev/null 2>&1; then
    echo "Error: Server not responding"
    exit 1
fi

# Get current count
CURRENT=$(curl -s "http://localhost:8080/statistics/database" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_vectors', 0))" 2>/dev/null || echo "0")
echo "Current vectors in DB: $CURRENT"
echo ""

START_TIME=$(python3 -c "import time; print(time.time())")
INSERTED=0
# Use timestamp to ensure unique keys
TIMESTAMP=$(date +%s)
START_KEY="${TIMESTAMP}_"

while [ $INSERTED -lt $TOTAL ]; do
    REMAINING=$((TOTAL - INSERTED))
    BATCH=$((REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE))
    
    # Generate batch using Python and write to temp file (avoids arg limit)
    TMPFILE=$(mktemp)
    python3 -c "
import random
import json

prefix = '$START_KEY'
offset = $INSERTED
batch_size = $BATCH
dims = $DIMS

keys = []
vectors = []
for i in range(batch_size):
    keys.append(f'vec_{prefix}{offset + i}')
    vectors.append([round(random.uniform(-1, 1), 6) for _ in range(dims)])

print(json.dumps({'keys': keys, 'vectors': vectors}))
" > "$TMPFILE"
    
    # Insert batch using file
    RESULT=$(curl -s -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d @"$TMPFILE")
    
    rm -f "$TMPFILE"
    
    INSERTED=$((INSERTED + BATCH))
    
    # Progress
    PCT=$((INSERTED * 100 / TOTAL))
    echo -ne "\rInserted: $INSERTED / $TOTAL ($PCT%)"
done

END_TIME=$(python3 -c "import time; print(time.time())")
DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
RATE=$(python3 -c "print(f'{$TOTAL / ($END_TIME - $START_TIME):.0f}')")

echo ""
echo ""
echo "================================================"
echo "Insert Complete!"
echo "================================================"
echo "Inserted: $TOTAL vectors"
echo "Time: ${DURATION}s"
echo "Rate: $RATE vectors/second"
echo ""

# Show new total
NEW_TOTAL=$(curl -s "http://localhost:8080/statistics/database" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_vectors', 0))" 2>/dev/null || echo "?")
echo "New total in DB: $NEW_TOTAL"

