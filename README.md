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
python src/prior_eval_build_all.py # every named public pair (765) for Table 1's top rows
python src/make_anchors.py        # the eight calibration examples -> submission/anchors.json
python src/disjoint_prep.py       # item files for the benchmarks used in no development step (ids in data/llm/item_ids_disjoint.json)
python reproduce.py               # all tables, from cached model outputs and the simulation logs in data/
python reproduce.py --rerun       # rerun every simulation first (JOBS=4 parallel jobs by default, several hours on a laptop), no API key
```

`--rerun` needs no API key on any platform. The simulated draws are replayed from `data/draws/` (benchmark, subject, item and row identifiers only, no responses), the subject priors from `data/llm/prior_cache.jsonl`, and `dl.py` pins the measurement-db revision the rows refer to. Three of these jobs, replayed on Linux from the same `data/long.parquet`, matched the macOS logs to within 0.00004 per draw. A checksum of the ordered table (`data/draws/long_parquet.sha256`) stops the replay if a table rebuilt by `dl.py` and `src/assemble.py` differs; a full rebuild on another machine has not been tested. The pooling simulations of Tables 5-6 (`src/sim_pp.py`) use no subject prior and draw afresh, so they match the logs up to sampling.

measurement-db is gated, so this repository ships item and subject identifiers, never item text or per-model results, and the scripts above rebuild them once access is granted. Language-model outputs are cached by item or pair: subject-prior estimates in `data/llm/prior_eval_*.json` and `data/llm/prior_cache.jsonl`, item ratings in `data/llm/api_haiku/` and `data/llm/batches_opus/`. `data/llm/prior_estimates.csv` lists the Opus, Sonnet and Haiku accuracy estimates for all 87 training pairs. `data/forecasts/forecasts.csv` gathers every accuracy forecast of the report in one table, 757 pairs from the main evaluation and 400 from the new benchmarks, and `src/score_forecasts.py` scores it or any other forecaster on the same pairs (`data/forecasts/README.md`). The online tables need the Codabench scoring logs of the listed runs (`src/online_table.py`), which are private to the submitting team, so `reproduce.py` skips them when `data/online/` is absent.

| Report | Source |
|---|---|
| Table 1 (first and last blocks), Tables 3 and 4, recall tests: subject prior | `reproduce.py` from `data/llm/prior_eval_*.json` (estimates made with `src/prior_eval_deployed.py`; all 765 public pairs in `data/llm/prior_estimates_all.csv`). The true accuracies come from measurement-db, so the scores need access. |
| Table 1 (middle block) and Figure 2: the 44 newer benchmarks | forecasts in `data/forecasts/forecasts.csv` (`set = new`), scored by `src/score_forecasts.py`; simulations by `src/sim_extra_build.py` and `src/sim_extra.py` on the draws in `data/llm/extra/sim/`. Both need measurement-db, and the per-pair outputs are not shipped. Figure 2 (`report/fig_gap.py`) plots the per-budget means of those runs; its true-accuracy line comes from the same loop with each pair's true accuracy as the prior. |
| Table 2: the submitted predictor with each language-model part switched off | `reproduce.py` from the logs in `data/ablate/` (`src/sim_sh.py`, subject priors replayed from `data/llm/prior_cache.jsonl`) |
| Table 2 caption, labels counted out of 31 | `tools/sim_acq31.py` (logs in `data/acq31/`) |
| Table 5 (Appendix B) and the four-subject result | `reproduce.py` from the logs in `data/disjoint/` and `data/multi/` (`src/sim_sh.py` with `ITEMS=data/llm/items_disjoint.json` or `NSUBS=4`) |
| Figure 1, Section 6 and Table 6: online runs | the Codabench scoring logs of the listed runs (`src/online_table.py`, `src/online_pairs.py`). They are private to the submitting team and not released, so `reproduce.py` skips this part when `data/online/` is absent. With the logs it prints the distinct-pair result of Section 6 (39 pairs, B0 0.199, lower on 33). |
| Table 7: variants that did not help | descriptions only; the exploratory scripts are in `src/explore/` |

Checks on systems released after a predictor's training data (Section 7 of the report) need the API or the Claude Code CLI and the gated measurement-db: `src/prior_eval_unseen.py` with `src/prior_unseen_analysis.py` and `src/prior_unseen_control.py` (Claude Opus 4.5; frozen pair lists and hashes in `data/llm/unseen_manifest/`), and `src/prior_eval_chibench.py` (the deployed prior on CHI-Bench). Their outputs contain true accuracies and are not shipped. Before the CHI-Bench run, one engineering probe of the CLI route was made on one target (Kimi K3); the reported analysis uses the fresh results of the full run, not a choice between the two.

Item ratings were produced by `src/rate_prep.py` (prompts) with `src/rate_api.py` (Haiku) and `src/run_rate_opus.sh` (Opus). `src/explore/` holds the exploratory scripts behind the dropped variants of Table 8. No table depends on them.

## The submission

`submission/model.py` (predictor) and `submission/labeling.py` (acquisition). `model.py` is the predictor of Codabench submission 1017682 (named v26 in the logs), except that it loads the calibration examples from `anchors.json` instead of carrying them inline. `model_v5.py` and `model_v6.py` are the earlier pooled predictors in the simulation tables. Python standard library and NumPy only. No parameters are fitted to the training responses; the prior's calibration examples and constants and the simulation-tuned settings are listed in Section 3 of the report.

- Without an API key the language-model terms are skipped and the pooled predictor runs on its own.
- With `llm_config.json` next to `model.py` (see `submission/llm_config.example.json`), it asks Claude Opus 5.5 for each subject's accuracy (once per subject and benchmark) and each item's difficulty. Subject-prior calls fall back to Claude Sonnet 5.5 and then Claude Haiku 5.5 when Opus gives no number. Results are cached on disk so the platform's worker processes share them, calls retry on rate limits, and per-process caps limit logical calls (150 prior calls, 300 item ratings). These are not dollar limits: checkpoints restart workers, each worker has its own allowance, retries can add HTTP attempts, and token use varies. Do not assume $10–15 covers a run.

Build the Codabench zip with `tools/make_zip.sh submission/model.py KEYFILE NAME` and check it with `check_submission_zip.py --static-only` from `aims-foundations/paiec_baseline`.

## Disclosures

Training data: public measurement-db only, for development, simulation and the eight calibration examples; no parameters are fitted to it. Pretrained models: Claude Opus 5.5 via the Anthropic API for subject priors and item ratings, with Claude Sonnet 5.5 and Claude Haiku 5.5 as fallbacks. Development used LLM coding assistants (Claude Code, Codex).

### API prefix cache for the current TINIT candidate (2026-11-04)

`dist/tinit1_cached/pkg/` and `dist/tinit1_cached/tinit1_cached_no_credentials.zip`
are the current **cache-enabled candidate**, derived from the TDEPLOY3 original
package with `tools/build_tinit1_cached.py TDEPLOY3_COPY DESTINATION`.
They contain no credentials and have not been uploaded. Weights, labeling,
concurrency and prediction rules are unchanged; the manifest binds both versions.
The old sealed experiment packages remain historical evidence.

- Prior requests split the fixed calibration examples from variable items and
  subject details. An explicit `cache_control: {"type": "ephemeral"}` breakpoint
  follows the shared examples, with the default 5-minute TTL. Identical prefixes
  can then be reused within each model. Prompt text and order are preserved.
- Rating requests mark the fixed system prefix too, but that short prefix is
  currently below the model's minimum cache length. **Do not count this as an
  effective rating-cache hit or predict savings on those requests.** We do not
  pad the prompt or cache every changing item merely to advertise caching.
- The generic `tools/make_zip.sh` applies the same transformation before it reads
  credentials. Unrecognized prompt construction fails for review rather than
  silently shipping an uncached variant.
- Paid-run preparation must inspect the actual outgoing request and distinguish
  local result reuse from API prefix caching. After execution report
  `cache_creation_input_tokens`, `cache_read_input_tokens` and hit requests.
  A cache marker is not evidence of a provider hit. TLIVE1 accounting now reserves
  for the higher first-write rate and reports cache usage explicitly.
- No additional paid call was made to validate this change. No-network tests
  cover prompt preservation, stable breakpoints, fallback, cache write/read
  pricing, usage reporting and the generated package. The old TLIVE1 source/seal
  remains the baseline; its old execution authorization is exhausted.

Provider reference: [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
A first write has a premium; subsequent hits save input cost. Cache reuse and
actual savings must be confirmed from usage, not inferred from request flags.
