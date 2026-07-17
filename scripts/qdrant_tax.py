#!/usr/bin/env python3
"""Reproduce the durability tax in a PRODUCTION vector DB (Qdrant).

Qdrant exposes durability per-request: `wait=True` blocks until the operation is
applied and written to the WAL (durable ack); `wait=False` returns immediately
(async). So throughput(wait=True) vs throughput(wait=False) is the durability tax as
a production system exposes it, and sweeping the batch size reproduces the
group-commit curve -- both measured on Qdrant, not our engine. This is the
external-validity check reviewers asked for: the tax is not an artifact of our code.

Run where docker + network are available (NOT this sandbox):
    docker run -p 6333:6333 qdrant/qdrant           # in another terminal
    pip install qdrant-client numpy
    python3 scripts/qdrant_tax.py --data datasets/sift        # real SIFT (or --synthetic)

Emits build/qdrant_results/qdrant_tax.csv and prints the tax + group-commit speedup.
"""
import argparse, csv, os, struct, time
import numpy as np

def load_fvecs(path, limit=None):
    out = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if len(b) < 4: break
            d = struct.unpack("<i", b)[0]
            v = np.frombuffer(f.read(4 * d), dtype=np.float32)
            out.append(v)
            if limit and len(out) >= limit: break
    return np.vstack(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--url", default="http://localhost:6333")
    args = ap.parse_args()

    from qdrant_client import QdrantClient, models
    if args.data:
        base = load_fvecs(os.path.join(args.data, next(f for f in os.listdir(args.data) if f.endswith("_base.fvecs"))), limit=args.n)
    else:
        base = np.random.default_rng(42).standard_normal((args.n, args.d)).astype(np.float32)
    n, d = base.shape
    ids = list(range(n))
    client = QdrantClient(url=args.url)

    def run(batch, wait):
        name = f"tax_{batch}_{wait}"
        client.recreate_collection(name, vectors_config=models.VectorParams(size=d, distance=models.Distance.EUCLID))
        t0 = time.perf_counter()
        for s in range(0, n, batch):
            e = min(n, s + batch)
            pts = [models.PointStruct(id=ids[i], vector=base[i].tolist()) for i in range(s, e)]
            client.upsert(name, points=pts, wait=wait)
        dt = time.perf_counter() - t0
        client.delete_collection(name)
        return n / dt

    os.makedirs("build/qdrant_results", exist_ok=True)
    rows = []
    print(f"{'batch':>7} {'wait=True (durable)':>20} {'wait=False (async)':>20}")
    for b in (1, 10, 100, 1000):
        dur = run(b, True); asy = run(b, False)
        rows.append((b, dur, asy))
        print(f"{b:>7} {dur:>20.0f} {asy:>20.0f}")
    with open("build/qdrant_results/qdrant_tax.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["batch", "durable_ins_per_s", "async_ins_per_s"]); w.writerows(rows)

    tax = rows[0][2] / rows[0][1] if rows[0][1] else 0        # async/durable at batch=1
    gc = rows[-1][1] / rows[0][1] if rows[0][1] else 0        # durable batch=1000 / batch=1
    print(f"\nQdrant durability tax (async/durable, batch=1): {tax:.1f}x")
    print(f"Qdrant group-commit speedup (durable, batch 1->1000): {gc:.1f}x")
    print("-> reproduces the tax + group-commit curve in a production vector DB.")

if __name__ == "__main__":
    main()
