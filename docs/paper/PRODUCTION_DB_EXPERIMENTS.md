# Production vector-DB reproduction (Lever 3+ / external validity)

Goal: show the durability tax and the group-commit curve are **not artifacts of our
engine** by reproducing them in a production system (Qdrant). This is the strongest
generality claim available short of re-implementing several systems.

## Why Qdrant
Qdrant exposes durability per request: `upsert(..., wait=True)` blocks until the op
is applied and in the WAL (durable ack); `wait=False` returns immediately. So:
- **tax** = throughput(`wait=False`) / throughput(`wait=True`) at batch=1;
- **group commit** = throughput(`wait=True`) as batch size grows.
Both are the same phenomena the paper isolates in our engine, measured on a widely
deployed vector DB.

## Run (needs docker + network; not this sandbox)
```
docker run -p 6333:6333 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant
pip install qdrant-client numpy
python3 scripts/qdrant_tax.py --data datasets/sift        # or --synthetic --n 20000
```
Output: `build/qdrant_results/qdrant_tax.csv` + printed tax and group-commit speedup.

## Fold into the paper
Add a short paragraph to §5 (Group Commit) or §8 (Related Work / generality):
"To confirm the tax is not specific to our implementation, we measured Qdrant: at
batch=1, durable (`wait=True`) upserts run at X/s vs Y/s async (a Zx tax), and
batching to 1000 recovers Wx -- the same shape as Fig.~\ref{fig:gc}." Cite Qdrant.

## Optional: Milvus
Milvus acks after WAL append; its durability/flush knobs (`flush()`, WAL config) give
an analogous knob. Same protocol, more setup. Qdrant is the cleanest single data point.

## Note on comparability
This measures each system's *own* durability knob, not a like-for-like fsync. Report
it as "the tax phenomenon reproduces" (qualitative shape + order of magnitude), not
as a head-to-head throughput comparison against our engine.
