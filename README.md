# Pooled Priors (PAIEC 2026)

Submission, simulator and cached model outputs for the Predictive AI Evaluation Competition. Technical report: `report/main.pdf`. Final Codabench submission: 1017682.

Before a model has answered anything on a new benchmark, we ask a language model how often that model answers items like these. On held-out training benchmarks the deployed estimate is worth about six of the model's own answers. It becomes a Beta prior inside a predictor that pools revealed labels across models and calibrates per-item difficulty ratings on the fly.

## Reproduce every table, no API key

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scipy huggingface_hub matplotlib
python dl.py                      # public measurement-db -> data/mdb (gated: request access on Hugging Face first)
python src/assemble.py            # binary-response benchmarks -> data/long.parquet
python src/rate_prep.py           # item files for the 12 simulation benchmarks (ids in data/llm/item_ids.json)
python src/prior_eval_build.py    # the 87 training pairs for the subject-prior evaluation
python src/make_anchors.py        # the eight calibration examples -> submission/anchors.json
python reproduce.py               # all tables, from cached model outputs and the simulation logs in data/
python reproduce.py --rerun       # rerun every simulation first (JOBS=6 parallel jobs, several hours on a laptop), still no API key
```

measurement-db is gated, so this repository ships item and subject identifiers, never item text or per-model results, and the scripts above rebuild them once access is granted. Language-model outputs are cached by item or pair: subject-prior estimates in `data/llm/prior_eval_*.json` and `data/llm/prior_cache.jsonl`, item ratings in `data/llm/api_haiku/` and `data/llm/batches_opus/`. `data/llm/prior_estimates.csv` lists the Opus, Sonnet and Haiku accuracy estimates for all 87 training pairs. The online tables need the Codabench scoring logs of the listed runs (`src/online_table.py`), which are private to the submitting team, so `reproduce.py` skips them when `data/online/` is absent.

| Report | Source |
|---|---|
| Table 1 and the recall tests, subject prior | `reproduce.py` from `data/llm/prior_eval_*.json` (estimates made with `src/prior_eval_deployed.py`) |
| Tables 2-3, online runs | `src/online_table.py` (run IDs inside) |
| Table 4, submitted predictor with each language-model term switched off | `src/sim_sh.py`, subject priors replayed from `data/llm/prior_cache.jsonl` (logs in `data/ablate/`) |
| Table 5 and Table A1, pooling in simulation | `src/sim_pp.py` (commands in `reproduce.py`, logs in `data/` and `data/small/`) |
| Figure A1 | `report/fig_budget.py` |
| Item-rating signal given the true pair accuracy (Section 3) | `src/item_signal_oracle.py`, log in `data/item_signal_oracle.log` |

## The submission

`submission/model.py` (predictor) and `submission/labeling.py` (acquisition). `model.py` is the predictor of Codabench submission 1017682 (named v26 in the logs), except that it loads the calibration examples from `anchors.json` instead of carrying them inline. `model_v5.py` and `model_v6.py` are the earlier pooled predictors in the simulation tables. Python standard library and NumPy only, no parameters fitted to the training data.

- Without an API key the language-model terms are skipped and the pooled predictor runs on its own.
- With `llm_config.json` next to `model.py` (see `submission/llm_config.example.json`), it asks Claude Opus 5.5 for each subject's accuracy (once per subject and benchmark) and each item's difficulty, falling back to Claude Sonnet 5.5 and Claude Haiku 5.5 when a call is refused. Results are cached on disk so the platform's worker processes share them, calls retry on rate limits, and per-process caps (150 prior calls, 300 item ratings) bound the cost to about $10-15 per formative run.

Build the Codabench zip with `tools/make_zip.sh submission/model.py KEYFILE NAME` and check it with `check_submission_zip.py --static-only` from `aims-foundations/paiec_baseline`.

## Disclosures

Training data: public measurement-db only, for development, simulation and the eight calibration examples; nothing is fitted to it. Pretrained models: Claude Opus 5.5 via the Anthropic API for subject priors and item ratings, with Claude Sonnet 5.5 and Claude Haiku 5.5 as fallbacks. Development used LLM coding assistants (Claude Code, Codex).
