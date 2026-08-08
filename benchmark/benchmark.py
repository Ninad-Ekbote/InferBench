"""Entry point that orchestrates a full benchmark run against a vLLM server.

Spins up `--num-users` concurrent workers that pull from a shared queue of
`--num-requests` jobs, so the concurrency level and total request count are
independent knobs. Each worker sends one streaming request at a time
through `client.VLLMClient`, appends the resulting `RequestMetrics` to a
shared list, and moves on to the next queued request.

Usage:
    python benchmark.py --server-url http://<runpod-host>:8000 --model my-model \\
        --num-users 20 --num-requests 200 --max-tokens 256 --run-name baseline

Re-running with different `--tag`/`--run-name` values (e.g. after enabling
prefix caching or chunked prefill on the server) requires no code changes —
only the server-side configuration under test changes between runs.
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from client import VLLMClient
from config import BenchmarkConfig, parse_args
from metrics import RequestMetrics, print_summary, summarize, write_csv
from utils import adjust_prompt_length, load_prompts, setup_logging

logger = logging.getLogger("benchmark.run")


async def _worker(
    worker_id: int,
    client: VLLMClient,
    prompts: list[str],
    request_queue: "asyncio.Queue[int]",
    results: list[RequestMetrics],
) -> None:
    """Drain `request_queue`, sending one request at a time until it's empty."""
    while True:
        try:
            request_id = request_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        prompt_index = request_id % len(prompts)
        prompt = prompts[prompt_index]
        result = await client.send_request(request_id, worker_id, prompt_index, prompt)
        results.append(result)
        request_queue.task_done()

        logger.info(
            "req=%-5d user=%-3d success=%s ttft=%s tokens=%d",
            request_id,
            worker_id,
            result.success,
            f"{result.ttft * 1000:.1f}ms" if result.ttft is not None else "n/a",
            result.tokens_generated,
        )
        if not result.success:
            logger.warning("req=%d error: %s", request_id, result.error)


async def run_benchmark(config: BenchmarkConfig) -> None:
    """Run the full benchmark described by `config` and print/save its results."""
    raw_prompts = load_prompts(config.prompts_file)
    prompts = [adjust_prompt_length(p, config.prompt_length) for p in raw_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), config.prompts_file)

    queue: "asyncio.Queue[int]" = asyncio.Queue()
    for request_id in range(config.num_requests):
        queue.put_nowait(request_id)

    results: list[RequestMetrics] = []
    connector = aiohttp.TCPConnector(limit=config.num_users)

    logger.info(
        "Starting benchmark: users=%d requests=%d model=%s server=%s tags=%s",
        config.num_users, config.num_requests, config.model, config.server_url, config.tags,
    )

    start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        client = VLLMClient(config, session)
        workers = [
            _worker(worker_id, client, prompts, queue, results)
            for worker_id in range(config.num_users)
        ]
        await asyncio.gather(*workers)
    wall_time = time.perf_counter() - start

    results.sort(key=lambda r: r.request_id)
    write_csv(results, config.output_csv)

    summary = summarize(results, wall_time)
    print_summary(summary)


def main() -> None:
    config = parse_args()
    setup_logging(config.log_level)
    logger.debug("Config: %s", config)
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
