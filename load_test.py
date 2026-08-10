import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from env import load_env

RESULTS_FILE = "results/load_test.csv"
PROMPTS_FILE = "prompts.json"


def send_request(host, port, model, prompt, max_tokens):
    url = f"http://{host}:{port}/v1/completions"
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        json.load(resp)
    return time.perf_counter() - start


def main():
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="omit to cycle through prompts.json")
    parser.add_argument("--users", type=int, default=10, help="number of concurrent requests")
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", 8000)))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt2"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as pool:
        latencies = list(
            pool.map(
                lambda i: send_request(
                    args.host, args.port, args.model, prompts[i % len(prompts)], args.max_tokens
                ),
                range(args.users),
            )
        )
    total = time.perf_counter() - start
    avg_latency = sum(latencies) / len(latencies)
    throughput = args.users / total

    print(f"{args.users} concurrent requests in {total:.2f}s")
    print(f"avg latency: {avg_latency:.2f}s")
    print(f"throughput: {throughput:.2f} req/s")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["timestamp", "model", "users", "max_tokens", "total_s", "avg_latency_s", "throughput_req_s"]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                args.model,
                args.users,
                args.max_tokens,
                f"{total:.4f}",
                f"{avg_latency:.4f}",
                f"{throughput:.4f}",
            ]
        )


if __name__ == "__main__":
    main()
