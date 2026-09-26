"""Offline ML prep for the adverse-action notice generator (Phase 3).

Downloads the UCI German Credit dataset, trains a small logistic regression,
selects 4-5 diverse sample applicants, computes SHAP values, and writes
~/workspace/rai-portfolio/work/shap_adverse_action.json
"""
import json
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
COLS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_length", "installment_rate", "personal_status",
    "other_debtors", "residence_years", "property_magnitude", "age", "other_installment",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
    "target_raw",
]

FEATURE_DEFS = {
    "duration_months": "Loan duration in months",
    "credit_amount": "Requested credit amount (DM)",
    "installment_rate": "Installment rate as a percentage of disposable income",
    "residence_years": "Years at current residence",
    "age": "Age in years",
    "existing_credits": "Number of existing credits at this bank",
    "people_liable": "Number of people being liable to provide maintenance",
    "checking_status_code": "Checking account status (0=no account, 1=balance <0, 2=balance 0-200, 3=balance >=200 or salary >=1yr)",
    "credit_history_code": "Credit history (0=bad/critical, 1=good, 2=very good)",
    "savings_status_code": "Savings account status (0=no account, 1=<100, 2=100-500, 3=500-1000, 4=>=1000 DM)",
    "employment_length_code": "Employment length (0=unemployed, 1=<1yr, 2=1-4yr, 3=4-7yr, 4=>=7yr)",
    "housing_code": "Housing (0=rent, 1=own, 2=free)",
    "job_code": "Job type (0=unemployed/unskilled non-resident, 1=unskilled resident, 2=skilled, 3=management/self-employed/highly qualified)",
}

# Human-meaningful notice features mapped from raw German Credit columns.
# Each entry: (new_name, source_column, transform)
# The German Credit dataset is ordered categorical codes; we map them to
# numeric proxies for the notice features the page will explain.

def map_row(r):
    chk = r["checking_status"]        # A11..A14
    cred_hist = r["credit_history"]   # A30..A34
    sav = r["savings_status"]         # A61..A65
    emp = r["employment_length"]      # A71..A75
    housing = r["housing"]            # A151..A153
    job = r["job"]                    # A171..A174
    chk_code = {"A11": 0, "A12": 1, "A13": 2, "A14": 3}[chk]
    hist_code = {"A30": 1, "A31": 2, "A32": 0, "A33": 1, "A34": 2}[cred_hist]
    sav_code = {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 4}[sav]
    emp_code = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}[emp]
    housing_code = {"A151": 0, "A152": 1, "A153": 2}[housing]
    job_code = {"A171": 0, "A172": 1, "A173": 2, "A174": 3}[job]
    # Deterministic pseudo-features derived from codes (seeded per row for reproducibility).
    # Each plain-language notice feature is a tight, monotone proxy of one German Credit
    # code so the model learns through the human-meaningful features, not the raw codes.
    import hashlib
    seed = int(hashlib.md5("|".join(str(v) for v in r.values).encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    util = ({0: 0.88, 1: 0.70, 2: 0.42, 3: 0.22}[chk_code]
            + rng.uniform(-0.04, 0.04))
    util = float(np.clip(util, 0.02, 0.99))
    late_base = {0: 4.5, 1: 0.8, 2: 0.15}[hist_code]
    late = int(np.clip(rng.poisson(late_base), 0, 9))
    dti = float(np.clip(r["installment_rate"] * 8.0 + 6.0
                        + rng.uniform(-3, 3), 5.0, 55.0))
    income = float(28000 + job_code * 20000 + sav_code * 8000
                   + rng.uniform(-6000, 6000))
    income = float(np.clip(income, 20000, 160000))
    return {
        "credit_utilization_pct": round(util * 100, 1),
        "late_payments_30d": late,
        "debt_to_income_pct": round(dti, 1),
        "annual_income": round(income, 0),
        "employment_length_months": int([0, 6, 24, 60, 96][emp_code]),
        "loan_duration_months": int(r["duration_months"]),
        "housing_code": housing_code,
        "residence_years": int(r["residence_years"]),
        "age_years": int(r["age"]),
        "existing_credits": int(r["existing_credits"]),
        "people_liable": int(r["people_liable"]),
    }


def main():
    print("Downloading dataset...")
    data = urllib.request.urlopen(DATA_URL, timeout=60).read().decode()
    rows = [line.split() for line in data.strip().splitlines()]
    df = pd.DataFrame(rows, columns=COLS)
    for c in ["duration_months", "credit_amount", "installment_rate", "residence_years",
              "age", "existing_credits", "people_liable", "target_raw"]:
        df[c] = pd.to_numeric(df[c])
    # target: 1 = good (approved), 2 = bad (denied)
    df["target"] = (df["target_raw"] == 1).astype(int)
    print("rows:", len(df), "approval rate:", round(df.target.mean(), 3))

    feats = pd.DataFrame([map_row(r) for _, r in df.iterrows()])
    feature_names = list(feats.columns)
    X = feats.values.astype(float)
    y = df["target"].values

    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    acc = model.score(scaler.transform(X_test), y_test)
    print("test accuracy:", round(acc, 3))

    # SHAP with LinearExplainer on standardized features
    import shap
    explainer = shap.LinearExplainer(model, scaler.transform(X_train))
    shap_values = explainer.shap_values(scaler.transform(X_test))
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)

    # Pick diverse applicants from the TEST set: find candidates
    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    Xf = feats.iloc[idx_test].copy().reset_index(drop=True)
    proba_s = pd.Series(proba)
    pred = (proba >= 0.5).astype(int)

    # Pick diverse applicants from the TEST set.
    top3_sets = {i: set(sorted(range(len(feature_names)),
                               key=lambda k: -abs(shap_values[i][k]))[:3])
                 for i in range(len(X_test))}

    denied_idx = list(proba_s[pred == 0].index)
    approved_idx = list(proba_s[pred == 1].index)

    # Denied 1: lowest approval probability
    d1 = int(denied_idx[int(np.argmin(proba_s.loc[denied_idx]))])
    # Denied 2: the denied applicant whose top-3 reasons overlap LEAST with d1's
    top1 = {i: max(top3_sets[i], key=lambda k: abs(shap_values[i][k])) for i in denied_idx}
    cand = [i for i in denied_idx if top1[i] != top1[d1]]
    pool = cand if cand else denied_idx
    d2 = int(min(pool, key=lambda i: (len(top3_sets[i] & top3_sets[d1]),
                                      proba_s.loc[i])))

    # Approved 1: highest approval probability
    a1 = int(approved_idx[int(np.argmax(proba_s.loc[approved_idx]))])
    # Approved 2: approved, mid-range probability, reasons different from a1's
    mid = [i for i in approved_idx if 0.55 <= proba_s.loc[i] <= 0.8]
    if mid:
        a2 = int(min(mid, key=lambda i: len(top3_sets[i] & top3_sets[a1])))
    else:
        a2 = int(approved_idx[int(np.argmin(np.abs(proba_s.loc[approved_idx] - 0.65)))])

    # Name applicants by their dominant reason (direction-aware)
    REASON_NAMES = {
        "credit_utilization_pct": ("high_utilization", "low_utilization"),
        "late_payments_30d": ("recent_late_payments", "clean_payment_history"),
        "debt_to_income_pct": ("high_debt_to_income", "low_debt_to_income"),
        "annual_income": ("low_income", "high_income"),
        "employment_length_months": ("short_employment", "stable_employment"),
        "loan_duration_months": ("long_loan_duration", "short_loan_duration"),
        "age_years": ("young_age", "older_age"),
    }

    def label(i, outcome):
        sv = shap_values[i]
        if outcome == "denied":
            k = max(range(len(feature_names)), key=lambda k: -sv[k] if sv[k] < 0 else -1e9)
            name = REASON_NAMES.get(feature_names[k], (feature_names[k], feature_names[k]))[0]
        else:
            k = max(range(len(feature_names)), key=lambda k: sv[k] if sv[k] > 0 else -1e9)
            name = REASON_NAMES.get(feature_names[k], (feature_names[k], feature_names[k]))[1]
        return f"{outcome}_{name}"

    ids = [label(d1, "denied"), label(d2, "denied"),
           label(a1, "approved"), label(a2, "approved")]
    seen = {}
    for n in range(len(ids)):
        base = ids[n]
        suffix = 2
        while ids[n] in seen:
            ids[n] = f"{base}_{suffix}"
            suffix += 1
        seen[ids[n]] = True

    picks = list(zip(ids, [d1, d2, a1, a2]))

    applicants = []
    for label, i in picks:
        i = int(i)
        sv = shap_values[i]
        top3 = sorted(range(len(feature_names)), key=lambda k: -abs(sv[k]))[:3]
        applicants.append({
            "id": label,
            "prediction": int(pred[i]),
            "predicted_probability_approved": round(float(proba[i]), 4),
            "features": {f: (int(Xf[f][i]) if isinstance(Xf[f][i], (int, np.integer)) else float(Xf[f][i]))
                         for f in feature_names},
            "shap_values": {f: round(float(sv[k]), 4) for k, f in enumerate(feature_names)},
            "top_reasons": [{"feature": feature_names[k],
                             "shap_value": round(float(sv[k]), 4),
                             "value": (int(Xf[feature_names[k]][i]) if isinstance(Xf[feature_names[k]][i], (int, np.integer)) else float(Xf[feature_names[k]][i]))}
                            for k in top3],
        })

    out = {
        "dataset": {
            "name": "UCI Statlog German Credit (1000 rows)",
            "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "rows_used": len(df),
            "target": "1 = approved (good credit), 0 = denied (bad credit)",
            "note": "German Credit ordered categorical codes were mapped to human-meaningful notice features; revolving-utilization, late-payment, DTI and income are seeded deterministic proxies derived from the real codes.",
        },
        "model": {
            "type": "LogisticRegression (C=1.0), StandardScaler, features standardized",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_accuracy": round(float(acc), 4),
            "random_state": 42,
            "base_value_log_odds": round(float(explainer.expected_value), 4),
        },
        "features": [
            {"name": "credit_utilization_pct",
             "description": "Revolving credit utilization as a percentage of the credit limit"},
            {"name": "late_payments_30d",
             "description": "Number of payments 30 or more days past due in the last 24 months"},
            {"name": "debt_to_income_pct",
             "description": "Total monthly debt obligations as a percentage of monthly income"},
            {"name": "annual_income",
             "description": "Applicant annual income in Canadian dollars"},
            {"name": "employment_length_months",
             "description": "Time with current employer in months"},
            {"name": "loan_duration_months",
             "description": "Requested loan duration in months"},
            {"name": "housing_code",
             "description": "Housing: 0=rent, 1=own, 2=free"},
            {"name": "residence_years",
             "description": "Years at current residence"},
            {"name": "age_years",
             "description": "Applicant age in years"},
            {"name": "existing_credits",
             "description": "Number of existing credits at this bank"},
            {"name": "people_liable",
             "description": "Number of people the applicant is liable to support"},
        ],
        "global_feature_importance": [
            {"feature": feature_names[k], "mean_abs_shap": round(float(mean_abs[k]), 4)}
            for k in order
        ],
        "applicants": applicants,
    }

    import os
    os.makedirs(os.path.expanduser("~/workspace/rai-portfolio/work"), exist_ok=True)
    path = os.path.expanduser("~/workspace/rai-portfolio/work/shap_adverse_action.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)

    # Sanity check
    for a in applicants:
        if a["prediction"] == 0:
            print(a["id"], "top reasons:", [(r["feature"], r["shap_value"], r["value"]) for r in a["top_reasons"]])


if __name__ == "__main__":
    main()
