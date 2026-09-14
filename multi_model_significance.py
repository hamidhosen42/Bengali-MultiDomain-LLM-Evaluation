"""
Paired multi-model statistical analysis across all six LLMs.

For every dataset x prompting setting this script computes
  1. Cochran's Q omnibus test on per-sample correctness of the six models,
  2. post-hoc pairwise exact McNemar tests (15 model pairs) with Holm correction,
  3. within-model zero-shot vs few-shot exact McNemar tests (Holm-corrected over 24 tests),
  4. Holm-adjusted p-values for the existing Qwen-2.5-72B vs Gemma-2-27B permutation tests,
  5. 95% bootstrap confidence intervals of Macro-F1 for all six models
     (same procedure/seed as calculate_ci.py).

Outputs (Results/):
  cochran_q_results.csv, pairwise_mcnemar_holm.csv, fewshot_effect_mcnemar.csv,
  qwen_gemma_permutation_holm.csv, f1_confidence_intervals_all_models.csv
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest

RESULTS_DIR = "Results"
DATASETS = ["senti", "emo", "hate", "fake"]
SETTINGS = ["zero_shot", "few_shot"]
MODELS = ["llama32-3B", "mistral-7B", "deepseek-8B", "phi4-14B", "gemma2-27B", "qwen-72B"]
DATASET_NAME = {"senti": "Sentiment", "emo": "Emotion", "hate": "Hate Speech", "fake": "Fake News"}


def clean_label(label):
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def load(ds, setting):
    df = pd.read_excel(os.path.join(RESULTS_DIR, f"{ds}_{setting}.xlsx"))
    y = df["label"].apply(clean_label).values
    preds = {m: df[m].apply(clean_label).values for m in MODELS}
    return y, preds


def cochran_q(correct):
    """correct: (n_samples, k_models) boolean matrix."""
    x = correct.astype(int)
    k = x.shape[1]
    G = x.sum(axis=0)          # successes per model
    L = x.sum(axis=1)          # successes per sample
    num = (k - 1) * (k * np.sum(G ** 2) - np.sum(G) ** 2)
    den = k * np.sum(L) - np.sum(L ** 2)
    Q = num / den
    p = chi2.sf(Q, k - 1)
    return Q, k - 1, p


def mcnemar_exact(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    if b + c == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return b, c, min(1.0, p)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(y_true, y_pred, classes, n_iterations=1000, alpha=0.05):
    n = len(y_true)
    np.random.seed(42)
    scores = []
    for _ in range(n_iterations):
        idx = np.random.randint(0, n, size=n)
        scores.append(macro_f1(y_true[idx], y_pred[idx], classes))
    scores = np.sort(scores)
    return np.percentile(scores, alpha / 2 * 100), np.percentile(scores, (1 - alpha / 2) * 100)


def main():
    q_rows, pair_rows, ci_rows = [], [], []
    correctness = {}

    for ds in DATASETS:
        for setting in SETTINGS:
            y, preds = load(ds, setting)
            classes = np.unique(y)
            correct = np.column_stack([preds[m] == y for m in MODELS])
            correctness[(ds, setting)] = correct

            Q, df_, p = cochran_q(correct)
            q_rows.append({"Dataset": DATASET_NAME[ds], "Setting": setting, "N": len(y),
                           "Cochran_Q": Q, "df": df_, "p_value": p})

            fam = []
            for i, j in itertools.combinations(range(len(MODELS)), 2):
                b, c, pv = mcnemar_exact(correct[:, i], correct[:, j])
                fam.append({"Dataset": DATASET_NAME[ds], "Setting": setting,
                            "Model_A": MODELS[i], "Model_B": MODELS[j],
                            "Acc_A": correct[:, i].mean(), "Acc_B": correct[:, j].mean(),
                            "A_right_B_wrong": b, "A_wrong_B_right": c, "p_raw": pv})
            adj = holm([r["p_raw"] for r in fam])
            for r, a in zip(fam, adj):
                r["p_holm"] = a
                r["significant_holm_0.05"] = a < 0.05
            pair_rows.extend(fam)

            for m in MODELS:
                base = macro_f1(y, preds[m], classes)
                lo, hi = bootstrap_ci(y, preds[m], classes)
                ci_rows.append({"Dataset": DATASET_NAME[ds], "Setting": setting, "Model": m,
                                "Test_Size": len(y), "Macro_F1": base, "CI_Lower": lo, "CI_Upper": hi})

    # zero-shot vs few-shot within each model (paired on the same test items)
    fs_rows = []
    for ds in DATASETS:
        y0, p0 = load(ds, "zero_shot")
        y1, p1 = load(ds, "few_shot")
        assert np.array_equal(y0, y1), f"test sets differ for {ds}"
        classes = np.unique(y0)
        for m in MODELS:
            c0, c1 = p0[m] == y0, p1[m] == y1
            b, c, pv = mcnemar_exact(c1, c0)  # b: few-shot right & zero-shot wrong
            fs_rows.append({"Dataset": DATASET_NAME[ds], "Model": m,
                            "Acc_zero": c0.mean(), "Acc_few": c1.mean(),
                            "MacroF1_zero": macro_f1(y0, p0[m], classes),
                            "MacroF1_few": macro_f1(y1, p1[m], classes),
                            "few_right_zero_wrong": b, "few_wrong_zero_right": c, "p_raw": pv})
    adj = holm([r["p_raw"] for r in fs_rows])
    for r, a in zip(fs_rows, adj):
        r["p_holm"] = a
        r["significant_holm_0.05"] = a < 0.05

    # Holm adjustment of the existing Qwen vs Gemma permutation tests (16 p-values)
    raw = pd.read_excel(os.path.join(RESULTS_DIR, "significance_tests_raw.xlsx"))
    # p = 0 from 1000 permutations is reported as < 0.001; use 0.001 as its upper bound
    raw["Acc_p_used"] = raw["Acc_p_val"].clip(lower=0.001)
    raw["F1_p_used"] = raw["F1_p_val"].clip(lower=0.001)
    allp = np.concatenate([raw["Acc_p_used"].values, raw["F1_p_used"].values])
    adjp = holm(allp)
    raw["Acc_p_holm"] = adjp[: len(raw)]
    raw["F1_p_holm"] = adjp[len(raw):]

    pd.DataFrame(q_rows).to_csv(os.path.join(RESULTS_DIR, "cochran_q_results.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(RESULTS_DIR, "pairwise_mcnemar_holm.csv"), index=False)
    pd.DataFrame(fs_rows).to_csv(os.path.join(RESULTS_DIR, "fewshot_effect_mcnemar.csv"), index=False)
    raw.to_csv(os.path.join(RESULTS_DIR, "qwen_gemma_permutation_holm.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(RESULTS_DIR, "f1_confidence_intervals_all_models.csv"), index=False)

    print("=== Cochran's Q ===")
    print(pd.DataFrame(q_rows).to_string(index=False))
    print("\n=== Pairwise McNemar (Holm): number of significant pairs out of 15 ===")
    pr = pd.DataFrame(pair_rows)
    print(pr.groupby(["Dataset", "Setting"])["significant_holm_0.05"].sum().to_string())
    print("\n=== Few-shot effect (McNemar, Holm over 24 tests) ===")
    print(pd.DataFrame(fs_rows).to_string(index=False))
    print("\n=== Qwen vs Gemma permutation p-values with Holm adjustment ===")
    print(raw[["Dataset", "Setting", "Acc_p_val", "Acc_p_holm", "F1_p_val", "F1_p_holm"]].to_string(index=False))
    print("\n=== 95% bootstrap CI of Macro-F1, all models ===")
    print(pd.DataFrame(ci_rows).to_string(index=False))


if __name__ == "__main__":
    main()
