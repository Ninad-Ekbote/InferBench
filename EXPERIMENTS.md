# Experiments

Planned runs, using `load_test.py` against the vLLM server started by `serve.sh`. Each run appends a row to `results/load_test.csv`.

## 1. Concurrency scaling

Same model, same max-tokens, increasing users — see how latency/throughput scale with load.

- `python load_test.py --users 5`
- `python load_test.py --users 10`
- `python load_test.py --users 20`
- `python load_test.py --users 50`

## 2. Max-tokens scaling

Fixed concurrency, increasing generation length.

- `python load_test.py --users 20 --max-tokens 32`
- `python load_test.py --users 20 --max-tokens 64`
- `python load_test.py --users 20 --max-tokens 128`
- `python load_test.py --users 20 --max-tokens 256`

## 3. Model comparison

Same GPU (RTX 5090), same users/max-tokens, different models. Update `MODEL_NAME` in `.env` and restart `serve.sh` between runs.

- `mistralai/Mistral-7B-Instruct-v0.2`
- (TBD — add more models to compare)

## 4. Speculative decoding

Baseline vs. speculative decoding (n-gram method, no extra draft model needed), same model/users/max-tokens, to see the latency/throughput gain.

- Baseline: `./serve.sh`
- Speculative: `SPECULATIVE=1 ./serve.sh` (or set `SPECULATIVE=1` in `.env`)

Restart `serve.sh` between the two, then run the same `load_test.py` command against each and compare.

## Status

- [x] GPU confirmed: 1x RTX 5090 (32GB VRAM)
- [ ] Run 1: concurrency scaling
- [ ] Run 2: max-tokens scaling
- [ ] Run 3: model comparison
- [ ] Run 4: speculative decoding
