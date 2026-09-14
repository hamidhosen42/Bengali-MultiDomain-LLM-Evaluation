"""
Quantitative error-category analysis on the saved model predictions (Results/*.xlsx).

Computes, for every model and prompting setting:
  1. the share of each true->predicted confusion pair among all errors
     (Results/error_confusion_pairs.csv),
  2. hate speech: false negatives vs false positives and predicted-hate rate
     (Results/error_hate_fn_fp.csv),
  3. sentiment/emotion: error rate on texts that contain an explicit negation marker
     versus texts without one (Results/error_negation_proxy.csv),
  4. fake news: surface cues (length, digits, quotation marks) of fake articles that were
     missed (predicted real) versus fake articles that were detected
     (Results/error_fake_surface_cues.csv).
"""
import os
import re
import numpy as np
import pandas as pd

RESULTS_DIR = "Results"
MODELS = ["llama32-3B", "mistral-7B", "deepseek-8B", "phi4-14B", "gemma2-27B", "qwen-72B"]
SETTINGS = ["zero_shot", "few_shot"]

# explicit Bengali negation markers: standalone না / নেই / নয় / নাই / নহে, or verb forms ending in -নি (e.g. হয়নি, করিনি)
NEG_STANDALONE = {"না", "নেই", "নয়", "নাই", "নহে"}
NEG_SUFFIX_RE = re.compile(r"[ঀ-৿]{2,}নি$")


def clean(label):
    if pd.isna(label):
        return ""
    return str(label).strip().lower()


def has_negation(text):
    tokens = re.findall(r"[ঀ-৿]+", str(text))
    for t in tokens:
        if t in NEG_STANDALONE or NEG_SUFFIX_RE.match(t):
            return True
    return False


def main():
    pair_rows, hate_rows, neg_rows, fake_rows = [], [], [], []
    for ds in ["senti", "emo", "hate", "fake"]:
        for st in SETTINGS:
            df = pd.read_excel(os.path.join(RESULTS_DIR, f"{ds}_{st}.xlsx"))
            y = df["label"].apply(clean)
            for m in MODELS:
                p = df[m].apply(clean)
                err = y != p
                n_err = int(err.sum())
                # 1. confusion pairs
                pairs = pd.crosstab(y[err], p[err])
                for t in pairs.index:
                    for q in pairs.columns:
                        c = int(pairs.loc[t, q])
                        if c:
                            pair_rows.append({"Dataset": ds, "Setting": st, "Model": m, "True": t,
                                              "Predicted": q, "Count": c, "Share_of_errors": c / n_err,
                                              "N_errors": n_err, "N": len(df)})
                # 2. hate FN/FP
                if ds == "hate":
                    fn = int(((y == "hate") & (p != "hate")).sum())
                    fp = int(((y != "hate") & (p == "hate")).sum())
                    hate_rows.append({"Setting": st, "Model": m, "FN_hate_as_nothate": fn,
                                      "FP_nothate_as_hate": fp, "FN_share_of_errors": fn / n_err,
                                      "Predicted_hate_rate": float((p == "hate").mean()),
                                      "Invalid_output": int((~p.isin(["hate", "not-hate"])).sum())})
                # 3. negation proxy
                if ds in ("senti", "emo"):
                    neg = df["sentence"].apply(has_negation)
                    neg_rows.append({"Dataset": ds, "Setting": st, "Model": m,
                                     "N_neg": int(neg.sum()), "N_nonneg": int((~neg).sum()),
                                     "ErrRate_neg": float(err[neg].mean()),
                                     "ErrRate_nonneg": float(err[~neg].mean())})
                # 4. fake surface cues
                if ds == "fake":
                    fake = y == "fake"
                    missed = fake & (p == "real")
                    caught = fake & (p == "fake")
                    txt = df["sentence"].astype(str)
                    words = txt.str.split().str.len()
                    digits = txt.str.contains(r"[0-9০-৯]")
                    quotes = txt.str.contains(r"[\"“”‘’']")
                    fake_rows.append({"Setting": st, "Model": m, "N_fake": int(fake.sum()),
                                      "Missed_fake": int(missed.sum()), "Miss_rate": float(missed.sum() / fake.sum()),
                                      "Mean_words_missed": float(words[missed].mean()),
                                      "Mean_words_caught": float(words[caught].mean()),
                                      "Digit_share_missed": float(digits[missed].mean()),
                                      "Digit_share_caught": float(digits[caught].mean()),
                                      "Quote_share_missed": float(quotes[missed].mean()),
                                      "Quote_share_caught": float(quotes[caught].mean())})

    pd.DataFrame(pair_rows).to_csv(os.path.join(RESULTS_DIR, "error_confusion_pairs.csv"), index=False)
    pd.DataFrame(hate_rows).to_csv(os.path.join(RESULTS_DIR, "error_hate_fn_fp.csv"), index=False)
    pd.DataFrame(neg_rows).to_csv(os.path.join(RESULTS_DIR, "error_negation_proxy.csv"), index=False)
    pd.DataFrame(fake_rows).to_csv(os.path.join(RESULTS_DIR, "error_fake_surface_cues.csv"), index=False)

    pairs = pd.DataFrame(pair_rows)
    print("=== Top confusion pairs (few-shot), share of all errors ===")
    for ds in ["senti", "emo", "hate", "fake"]:
        sub = pairs[(pairs.Dataset == ds) & (pairs.Setting == "few_shot")]
        pooled = sub.groupby(["True", "Predicted"])["Count"].sum().sort_values(ascending=False)
        tot = pooled.sum()
        print(f"\n{ds} pooled over 6 models (N_errors={tot}):")
        print((pooled / tot).head(6).round(3).to_string())
        for m in ["gemma2-27B", "qwen-72B"]:
            s = sub[sub.Model == m].sort_values("Share_of_errors", ascending=False)
            print(f"  {m}: " + "; ".join(f"{t}->{q} {c} ({s_:.1%})" for t, q, c, s_ in zip(s["True"], s["Predicted"], s["Count"], s["Share_of_errors"])))
    print("\n=== Hate speech FN vs FP ===")
    print(pd.DataFrame(hate_rows).round(3).to_string(index=False))
    print("\n=== Negation proxy ===")
    print(pd.DataFrame(neg_rows).round(3).to_string(index=False))
    print("\n=== Fake news surface cues ===")
    print(pd.DataFrame(fake_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
