After the runs, one portability edit (no change to prompts, models or analysis): the run scripts import
submission/model_v26.py when present and otherwise submission/model.py (the same predictor as shipped in the
release), and stop unless the 8 calibration examples are built (src/make_anchors.py, see README).
Hashes of the run scripts as run -> after the edit:
  src/prior_eval_unseen.py    96ac6111 -> 20be6e31
  src/prior_eval_chibench.py  529dad3c -> dd98fe09
  src/prior_eval_extra.py     57112342 -> b8b9523a
