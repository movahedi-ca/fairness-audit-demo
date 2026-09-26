"""Offline ML computation for the fairness audit console (Phase 5 of the Responsible AI portfolio).

Loads UCI Statlog German Credit (1000 rows), trains logistic regression
(without the sensitive attribute), sweeps decision thresholds, applies
fairlearn's ThresholdOptimizer (demographic parity + equalized odds), and
writes ~/workspace/rai-portfolio/work/fairness_audit.json with fully
deterministic, seed-fixed numbers.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from fairlearn.postprocessing import ThresholdOptimizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shap_prep  # reuses COLS, DATA_URL, map_row exactly (read-only)

SEED = 42
OUT = os.path.expanduser("~/workspace/rai-portfolio/work/fairness_audit.json")
R = lambda x: round(float(x), 4)

# ---------------------------------------------------------------- dataset --
data = __import__("urllib.request", fromlist=["urlopen"]).urlopen(shap_prep.DATA_URL, timeout=60).read().decode()
rows = [line.split() for line in data.strip().splitlines()]
df = pd.DataFrame(rows, columns=shap_prep.COLS)
for c in ["duration_months", "credit_amount", "installment_rate", "residence_years",
          "age", "existing_credits", "people_liable", "target_raw"]:
    df[c] = pd.to_numeric(df[c])
df["target"] = (df["target_raw"] == 1).astype(int)  # 1 = good (approved), 0 = bad (denied)

# ----------------------------------------------------- sensitive attribute --
# personal_status codes in German Credit:
#   A91 = male   divorced/separated,  A92 = female divorced/separated/married,
#   A93 = male   single,              A94 = male   married/widowed,
#   A95 = female single
SEX_MAP = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
codes = set(df["personal_status"].unique())
assert codes <= set(SEX_MAP), df["personal_status"].unique()
assert {"A91", "A92", "A93", "A94"} <= codes
missing_codes = sorted(set(SEX_MAP) - codes)  # codes documented but absent from the data
df["sex"] = df["personal_status"].map(SEX_MAP)
assert not df["sex"].isna().any()

# ------------------------------------------------------------------ model --
feats = pd.DataFrame([shap_prep.map_row(r) for _, r in df.iterrows()])
X = feats.values.astype(float)          # 11 notice features; sex EXCLUDED (fairness-through-unawareness test)
y = df["target"].values
sex_all = df["sex"].values

idx = np.arange(len(X))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
sex_train, sex_test = sex_all[idx_train], sex_all[idx_test]

scaler = StandardScaler().fit(X_train)
model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
model.fit(scaler.transform(X_train), y_train)
proba_test = model.predict_proba(scaler.transform(X_test))[:, 1]
print("train rows:", len(X_train), "| test rows:", len(X_test),
      "| base acc @0.5:", R((proba_test >= 0.5).astype(int).__eq__(y_test).mean()))
print("test sex counts:", dict(zip(*np.unique(sex_test, return_counts=True))))
print("train sex counts:", dict(zip(*np.unique(sex_train, return_counts=True))))

# --------------------------------------------- per-group metrics function --
def group_stats(y_true, y_pred, groups):
    """Return dict group -> {n, tp, fp, tn, fn, selection_rate, tpr, fpr, accuracy}."""
    stats = {}
    for g in ("female", "male"):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tp, fp, tn, fn = int(tp), int(fp), int(tn), int(fn)
        n = int(m.sum())
        sr = (tp + fp) / n
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        stats[g] = {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "selection_rate": R(sr), "tpr": R(tpr), "fpr": R(fpr),
                    "accuracy": R((tp + tn) / n)}
    return stats

# --------------------------------------------- point evaluation function ----
# Privileged group fixed globally from the base operating point (threshold 0.5):
# whichever group has the higher selection rate there is "privileged".
base_pred = (proba_test >= 0.5).astype(int)
base_stats = group_stats(y_test, base_pred, sex_test)
PRIV = max(("female", "male"), key=lambda g: base_stats[g]["selection_rate"])
UNPRIV = "female" if PRIV == "male" else "male"
print("privileged:", PRIV, "| unprivileged:", UNPRIV)

def evaluate(y_pred, name, detail):
    stats = group_stats(y_test, y_pred, sex_test)
    total_tp = stats["female"]["tp"] + stats["male"]["tp"]
    total_tn = stats["female"]["tn"] + stats["male"]["tn"]
    total_fp = stats["female"]["fp"] + stats["male"]["fp"]
    total_fn = stats["female"]["fn"] + stats["male"]["fn"]
    acc = (total_tp + total_tn) / len(y_test)
    di = stats[UNPRIV]["selection_rate"] / stats[PRIV]["selection_rate"]
    eo = max(abs(stats[UNPRIV]["tpr"] - stats[PRIV]["tpr"]),
             abs(stats[UNPRIV]["fpr"] - stats[PRIV]["fpr"]))
    return {
        "name": name, "detail": detail,
        "accuracy": R(acc),
        "disparate_impact_ratio": R(di),
        "equalized_odds_difference": R(eo),
        "selection_rates": {g: stats[g]["selection_rate"] for g in ("female", "male")},
        "per_group": {g: {k: stats[g][k] for k in ("n", "tp", "fp", "tn", "fn",
                                                   "selection_rate", "tpr", "fpr", "accuracy")}
                      for g in ("female", "male")},
    }

points = []
# 1-5: threshold sweep on the base model
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    pred = (proba_test >= t).astype(int)
    points.append(evaluate(pred, "threshold_sweep",
                           {"decision_threshold": t}))

# 6-7: fairlearn ThresholdOptimizer mitigations (fit on train, applied to test)
for constraint in ("demographic_parity", "equalized_odds"):
    to = ThresholdOptimizer(estimator=model,
                            constraints=constraint,
                            objective="accuracy_score",
                            predict_method="predict_proba")
    to.fit(scaler.transform(X_train), y_train, sensitive_features=sex_train)
    pred = to.predict(scaler.transform(X_test), sensitive_features=sex_test,
                      random_state=SEED).astype(int)
    points.append(evaluate(pred, "threshold_optimizer",
                           {"constraint": constraint, "objective": "accuracy_score"}))

# ------------------------------------------------- consistency verification --
worst = 0.0
for p in points:
    s = p["per_group"]
    # accuracy from confusion matrices
    tp = s["female"]["tp"] + s["male"]["tp"]
    tn = s["female"]["tn"] + s["male"]["tn"]
    acc_ck = (tp + tn) / len(y_test)
    # DI ratio from stored selection rates
    di_ck = p["selection_rates"][UNPRIV] / p["selection_rates"][PRIV]
    # EO diff from stored rounded tpr/fpr
    eo_ck = max(abs(s[UNPRIV]["tpr"] - s[PRIV]["tpr"]),
                abs(s[UNPRIV]["fpr"] - s[PRIV]["fpr"]))
    # per-group matrix internal consistency (selection rate from counts)
    for g in ("female", "male"):
        g_ck = (s[g]["tp"] + s[g]["fp"]) / s[g]["n"]
        worst = max(worst, abs(g_ck - s[g]["selection_rate"]))
        assert (s[g]["tp"] + s[g]["fp"] + s[g]["tn"] + s[g]["fn"]) == s[g]["n"]
    worst = max(worst, abs(acc_ck - p["accuracy"]),
                abs(di_ck - p["disparate_impact_ratio"]),
                abs(eo_ck - p["equalized_odds_difference"]))
print("max consistency deviation across all checks:", worst)
assert worst <= 0.0002, "consistency check failed"

# ------------------------------------------------------------------ output --
out = {
    "dataset": {
        "name": "UCI Statlog German Credit",
        "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "rows_total": 1000,
        "target": "1 = approved (good credit), 0 = denied (bad credit)",
        "overall_approval_rate": R(float(y.mean())),
        "sex_distribution": {g: int((df['sex'] == g).sum()) for g in ("female", "male")},
        "feature_note": "The 11 human-meaningful notice features are seeded deterministic proxies "
                        "derived from the real German Credit codes (see shap_prep.map_row); "
                        "they are identical to the Phase 3 adverse-action build.",
    },
    "sensitive_attribute": {
        "name": "sex",
        "source_column": "personal_status",
        "mapping": {
            "A91": "male (divorced/separated)",
            "A92": "female (divorced/separated/married)",
            "A93": "male (single)",
            "A94": "male (married/widowed)",
            "A95": "female (single)",
        },
        "mapping_source": "Standard UCI German Credit personal_status code semantics, "
                          "as used in the fairlearn/aif360 literature on this dataset. "
                          "Note: code A95 (female single) is documented but absent from the "
                          "downloaded file; it would map to female if present.",
        "absent_codes": missing_codes,
        "why_chosen": "Sex is the canonical protected attribute for credit-lending fairness audits "
                      "(e.g. US ECOA) and the standard choice in published fairness work on German Credit. "
                      "Age binned at 25 was rejected: the under-25 subgroup is roughly half the size of the "
                      "female subgroup, which would make disparity estimates noisier on the 200-row test set.",
        "included_as_model_feature": False,
        "privileged_group": PRIV,
        "unprivileged_group": UNPRIV,
        "privilege_rule": "Group with the higher selection rate at the base operating point "
                          "(threshold 0.5) is designated privileged; applied uniformly to all points.",
    },
    "model": {
        "type": "LogisticRegression",
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs (sklearn default)",
        "scaling": "StandardScaler fit on train only",
        "features": list(feats.columns),
        "sensitive_attribute_included": False,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_split": "80/20 stratified on target",
        "random_state": SEED,
    },
    "seeds": {"numpy_sklearn_random_state": SEED, "split": SEED,
              "note": "map_row uses per-row deterministic hashing. ThresholdOptimizer.fit is "
                      "deterministic; its randomized-interpolation predict() was called with "
                      "random_state=42. No unseeded randomness anywhere; two reruns produced "
                      "byte-identical JSON."},
    "metrics_definitions": {
        "accuracy": "Fraction of correct predictions on the 200-row test set.",
        "disparate_impact_ratio": "selection_rate(unprivileged) / selection_rate(privileged). "
                                 "1.0 = perfect parity; the EEOC 4/5 rule flags values below 0.8.",
        "equalized_odds_difference": "max(|TPR_unpriv - TPR_priv|, |FPR_unpriv - FPR_priv|). 0.0 = equalized odds.",
        "selection_rate": "(TP + FP) / group n, i.e. fraction predicted 'approved'.",
        "tpr": "TP / (TP + FN).", "fpr": "FP / (FP + TN).",
        "rounding": "All floats rounded to 4 decimals.",
    },
    "caveats": [
        "Test set is only 200 rows; the female test subgroup is ~60 rows, so disparity "
        "estimates carry small-sample noise. Results demonstrate the tradeoff machinery, "
        "not a production-grade fairness claim.",
        "German Credit (1994) encodes dated gender/family categories; 'sex' here is a binary "
        "proxy derived from the personal_status code, not a rich demographic measure.",
        "ThresholdOptimizer fits group-specific thresholds on the training set and applies them "
        "to the test set; its points are post-processing mitigations of the same base model, "
        "not retrained models.",
        "aif360 0.6.1 was installed successfully; metrics were computed directly with "
        "sklearn/fairlearn primitives for full determinism and auditability.",
    ],
    "tradeoff_points": points,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", OUT, "with", len(points), "tradeoff points")

for p in points:
    print(p["name"], p["detail"],
          "| acc", p["accuracy"], "| DI", p["disparate_impact_ratio"],
          "| EO", p["equalized_odds_difference"],
          "| sr", p["selection_rates"])
