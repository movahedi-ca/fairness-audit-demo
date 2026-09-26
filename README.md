# Fairness audit demo (offline computation)

Offline numbers behind the [Fairness Audit Console](https://movahedi.ca/tools/fairness-audit/)
on movahedi.ca.

## What it does

`fairness_audit.py` loads the UCI Statlog German Credit dataset (1,000 rows),
trains a logistic regression **without** the sensitive attribute (sex, derived
from `personal_status` — a fairness-through-unawareness setup), and audits it:

- **Threshold sweep** (0.3 .. 0.7): accuracy, disparate-impact ratio,
  equalized-odds difference, per-group selection rates and confusion matrices
- **Fairlearn `ThresholdOptimizer`** under demographic-parity and
  equalized-odds constraints (random_state=42, deterministic)

Results (test set, n=200): at threshold 0.5, accuracy 0.7350, disparate-impact
ratio 0.9174 (female/male), equalized-odds difference 0.1300. The
equalized-odds mitigation reaches EO difference 0.0750 at accuracy 0.7600.

Caveat: only 60 women in the test set, so per-group rates carry noise.
Educational demo, not a compliance review.

## Run it

```bash
pip install -r requirements.txt
python fairness_audit.py   # writes fairness_audit.json
```

## Files

- `fairness_audit.py` — the audit script (imports `shap_prep.py` for the
  German Credit column layout and feature mapping)
- `shap_prep.py` — dataset loading and feature engineering (shared with the
  adverse-action SHAP demo)
- `fairness_audit.json` — full results, including per-point per-group
  confusion matrices
