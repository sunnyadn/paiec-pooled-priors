# Language-model forecasts of benchmark accuracy

`forecasts.csv` holds one row per named (benchmark, subject) pair from measurement-db. Each row has the estimate that Claude gave, with the deployed prompt, of the share of the benchmark's items the subject answers correctly. The prompt saw the subject's name, provider, release date, harness and reasoning effort, plus up to eight sample items.

- `set = main`: 757 pairs from the 32 benchmarks of the main evaluation (Table 1). The eight pairs that are calibration examples in the prompt are left out. The other 480 rows on those eight benchmarks are kept and marked `benchmark_has_calibration_example = True`. The scorer uses only the 277 rows marked `False`.
- `set = new`: 400 pairs from 44 benchmarks that no part of our development touched (Section 4). The run order was fixed in advance.
- `estimate` is the raw answer, before the shrinkage toward 0.5 used by the predictor. `answered_by` names the model that answered (Opus, or the Sonnet/Haiku fallback).

The file holds no item text and no true accuracies. With access to measurement-db, rebuild the accuracies with `src/prior_eval_build_all.py` (main) and `src/prior_extra_build.py` (new), then run `python src/score_forecasts.py`. It prints the RMSE of the raw and shrunk estimates and of the constant 0.5, for each set and pooled. Any other forecaster can be scored on the same 677 pairs: pass a CSV with columns `set`, `bench`, `subject_id` and `estimate`. The scored pairs always come from `forecasts.csv`, and the scorer stops on a missing or duplicate pair or an estimate outside [0, 1].
