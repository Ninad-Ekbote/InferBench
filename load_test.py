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
    """Sends one streaming completion request and measures real TTFT/ITL/token counts.

    Returns a dict; on failure, 'error' is set and token/timing fields are None.
    """
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    completion_tokens = prompt_tokens = total_tokens = 0
    error = None

    try:
        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
    except Exception as e:
        error = str(e)

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    itl = None
    if len(token_times) > 1:
        gaps = [b - a for a, b in zip(token_times, token_times[1:])]
        itl = sum(gaps) / len(gaps)

    return {
        "latency": end - start,
        "ttft": ttft,
        "itl": itl,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "error": error,
    }


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
        results = list(
            pool.map(
                lambda i: send_request(
                    args.host, args.port, args.model, prompts[i % len(prompts)], args.max_tokens
                ),
                range(args.users),
            )
        )
    total = time.perf_counter() - start

    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    total_completion_tokens = sum(r["completion_tokens"] for r in ok)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    total_tokens = sum(r["total_tokens"] for r in ok)
    actual_tokens_per_sec = total_completion_tokens / total if total > 0 else 0.0

    avg_latency = sum(r["latency"] for r in ok) / len(ok) if ok else 0.0
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    itls = [r["itl"] for r in ok if r["itl"] is not None]
    avg_itl = sum(itls) / len(itls) if itls else None
    throughput_req_s = args.users / total if total > 0 else 0.0

    print(f"{args.users} concurrent requests in {total:.2f}s ({len(errors)} errors)")
    print(f"avg latency: {avg_latency:.3f}s")
    print(f"avg TTFT:    {avg_ttft:.3f}s" if avg_ttft is not None else "avg TTFT:    n/a")
    print(f"avg ITL:     {avg_itl:.4f}s" if avg_itl is not None else "avg ITL:     n/a")
    print(f"requests/s:  {throughput_req_s:.2f}")
    print(f"completion tokens/s (actual): {actual_tokens_per_sec:.2f}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "users",
                    "max_tokens",
                    "total_s",
                    "avg_latency_s",
                    "avg_ttft_s",
                    "avg_itl_s",
                    "requests_per_s",
                    "completion_tokens",
                    "prompt_tokens",
                    "total_tokens",
                    "actual_tokens_per_sec",
                    "errors",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                args.model,
                args.users,
                args.max_tokens,
                f"{total:.4f}",
                f"{avg_latency:.4f}",
                f"{avg_ttft:.4f}" if avg_ttft is not None else "",
                f"{avg_itl:.5f}" if avg_itl is not None else "",
                f"{throughput_req_s:.4f}",
                total_completion_tokens,
                total_prompt_tokens,
                total_tokens,
                f"{actual_tokens_per_sec:.4f}",
                len(errors),
            ]
        )


if __name__ == "__main__":
    main()
