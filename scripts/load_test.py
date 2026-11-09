"""Simple load test: send N requests with C concurrent clients and report latency percentiles.

Usage:  python scripts/load_test.py --url http://localhost:8000 --requests 200 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

SAMPLES = [
    "Spotless, beautiful home and the host was wonderful. Would stay again!",
    "Nice location but the room was dusty and we had to pay extra for parking.",
    "Check-in was easy, the bed was comfortable and the kitchen had everything we needed.",
    "The wifi did not work and the AC was broken during a 100 degree weekend.",
    "Great value for the price, quiet neighborhood, close to downtown Dallas.",
]


async def run(url: str, n: int, concurrency: int, batch: int) -> dict:
    latencies, errors = [], 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        (await client.get("/health")).raise_for_status()

        async def one() -> None:
            nonlocal errors
            async with sem:
                body = {"texts": random.choices(SAMPLES, k=batch)}
                start = time.perf_counter()
                try:
                    r = await client.post("/predict", json=body)
                    r.raise_for_status()
                    latencies.append((time.perf_counter() - start) * 1000)
                except httpx.HTTPError:
                    errors += 1

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        wall = time.perf_counter() - start

    lat = np.array(latencies) if latencies else np.array([np.nan])
    return {
        "url": url, "requests": n, "concurrency": concurrency, "texts_per_request": batch,
        "errors": errors, "throughput_rps": round(len(latencies) / wall, 2),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1, help="texts per request")
    parser.add_argument("--out", default="reports/load_test.json")
    args = parser.parse_args()
    result = asyncio.run(run(args.url, args.requests, args.concurrency, args.batch))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
