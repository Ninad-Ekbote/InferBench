import argparse
import json
import os
import urllib.request

from env import load_env


def main():
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", 8000)))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt2"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/completions"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.load(resp)
    print(result["choices"][0]["text"])


if __name__ == "__main__":
    main()
