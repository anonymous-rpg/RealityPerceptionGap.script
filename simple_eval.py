#!/usr/bin/env python3
"""eval_simple.py — minimal evaluation against a 2- or 3-column prediction CSV.

Expected formats:

  Prediction CSV  (positional, header row is auto-detected if first row is non-numeric)
    column 0 : video_id  (any string; matches GT by exact equality)
    column 1 : pred_label  (accepts: 0 / 1   OR   "real" / "fake"   OR  "0"/"1")
    column 2 : score  (optional; if present, AUC and TPR@1%FPR are reported)

  Ground-truth CSV (same shape, only first two columns used)
    column 0 : video_id
    column 1 : label  (0 = real, 1 = fake; or "real"/"fake")

Metrics (positive class = fake):
  Always:
    ACC, MacroF1, TPR_fake, TNR_real, BalancedAcc, MCC
  Additionally when prediction CSV has a 3rd score column:
    AUC, TPR@1%FPR
  AUC's score-direction is auto-flipped if raw < 0.5.

Usage:
  python eval_simple.py --pred preds.csv --gt groundtruth.csv
  python eval_simple.py --pred preds.csv --gt groundtruth.csv --label MyModel
  python eval_simple.py --pred preds.csv --gt groundtruth.csv --out-csv all_runs.csv
"""


import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def _coerce_label(v) -> int | None:
    """Accept 0/1 (int or str) or fake/real (case-insensitive). Return 1 = fake, 0 = real, None on bad."""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("1", "fake", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "real", "false", "f", "no", "n"):
        return 0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except ValueError:
        pass
    return None


def _read_two_or_three_col(path: Path, has_score: bool | None = None) -> pd.DataFrame:
    """Read a CSV that has 2 or 3 positional columns. Returns DataFrame with
    columns ['video_id', 'label_raw'] or ['video_id', 'label_raw', 'score']."""
    df = pd.read_csv(path, header=0)
    if df.shape[1] < 2:
        sys.exit(f"{path}: expected at least 2 columns, got {df.shape[1]}")

    # If first-row of any column looks like data (not header), we treat it as headerless.
    # Heuristic: if header looks like 'col0' or 'Unnamed:', re-read with header=None.
    if all(str(c).startswith("Unnamed:") or str(c).strip() == "" for c in df.columns):
        df = pd.read_csv(path, header=None)

    out = pd.DataFrame()
    out["video_id"] = df.iloc[:, 0].astype(str).str.strip()
    out["label_raw"] = df.iloc[:, 1]
    if df.shape[1] >= 3:
        out["score"] = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    return out


def evaluate(pred_csv: Path, gt_csv: Path) -> dict:
    pred = _read_two_or_three_col(pred_csv)
    gt = _read_two_or_three_col(gt_csv)

    has_score = "score" in pred.columns

    # Coerce labels
    pred["pred"] = pred["label_raw"].apply(_coerce_label)
    gt["label"] = gt["label_raw"].apply(_coerce_label)

    n_pred = len(pred)
    n_gt = len(gt)

    pred_clean = pred.dropna(subset=["pred"])
    gt_clean = gt.dropna(subset=["label"])
    n_pred_valid_label = len(pred_clean)
    n_gt_valid_label = len(gt_clean)

    # Match by video_id
    merged = pred_clean.merge(gt_clean[["video_id", "label"]], on="video_id", how="inner")
    n_matched = len(merged)
    if n_matched == 0:
        sys.exit("No video_id matches between pred and gt CSVs.")

    if has_score:
        merged = merged.dropna(subset=["score"])

    y = merged["label"].astype(int).values
    yhat = merged["pred"].astype(int).values

    if y.sum() == 0 or (1 - y).sum() == 0:
        sys.exit("Matched set is single-class — cannot compute metrics.")

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "n_pred":               n_pred,
        "n_pred_valid_label":   n_pred_valid_label,
        "n_gt":                 n_gt,
        "n_gt_valid_label":     n_gt_valid_label,
        "n_matched":            n_matched,
        "n_fake":               int(y.sum()),
        "n_real":               int((1 - y).sum()),
        "has_score":            int(has_score),
        "score_flipped":        0,
        "ACC":                  float(accuracy_score(y, yhat)),
        "MacroF1":              float(f1_score(y, yhat, average="macro", zero_division=0)),
        "TPR_fake":             float(tp / (tp + fn) if (tp + fn) else 0.0),
        "TNR_real":             float(tn / (tn + fp) if (tn + fp) else 0.0),
        "BalancedAcc":          float(balanced_accuracy_score(y, yhat)),
        "MCC":                  float(matthews_corrcoef(y, yhat)),
    }

    if has_score:
        s = merged["score"].astype(float).values
        auc_raw = roc_auc_score(y, s)
        flipped = auc_raw < 0.5
        if flipped:
            s = -s
            auc = roc_auc_score(y, s)
        else:
            auc = auc_raw
        fpr, tpr, _ = roc_curve(y, s)
        out["AUC"] = float(auc)
        out["TPR@1%FPR"] = float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0
        out["score_flipped"] = int(flipped)
    else:
        out["AUC"] = None
        out["TPR@1%FPR"] = None

    return out


def print_report(label: str, r: dict):
    print(f"\n=== {label} ===")
    print(f"  matched : {r['n_matched']:>5d}  (pred={r['n_pred']}/{r['n_pred_valid_label']} valid, "
          f"gt={r['n_gt']}/{r['n_gt_valid_label']} valid)")
    print(f"  fake    : {r['n_fake']:>5d}")
    print(f"  real    : {r['n_real']:>5d}")
    print()
    label_metrics = ["ACC", "MacroF1", "TPR_fake", "TNR_real", "BalancedAcc", "MCC"]
    score_metrics = ["AUC", "TPR@1%FPR"]
    for m in label_metrics:
        print(f"  {m:<12s} : {r[m]:.4f}")
    if r["has_score"]:
        for m in score_metrics:
            print(f"  {m:<12s} : {r[m]:.4f}")
        if r["score_flipped"]:
            print(f"  (score sign was flipped: raw AUC was below 0.5)")
    else:
        print()
        print("  AUC          : N/A  (no 'score' column in prediction CSV)")
        print("  TPR@1%FPR    : N/A  (no 'score' column in prediction CSV)")
        print("  → To enable AUC + TPR@1%FPR, add a 3rd column with continuous scores.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, help="Prediction CSV (video_id, pred_label[, score])")
    ap.add_argument("--gt",   required=True, type=Path, help="Ground-truth CSV (video_id, label)")
    ap.add_argument("--label", default=None, help="Display name. Default: prediction filename stem.")
    ap.add_argument("--out-csv", default=None, type=Path, help="Append a row to this CSV for batch comparison.")
    args = ap.parse_args()

    label = args.label or args.pred.stem
    r = evaluate(args.pred, args.gt)
    print_report(label, r)

    if args.out_csv:
        cols = ["label", "ACC", "MacroF1", "AUC", "TPR_fake", "TNR_real", "TPR@1%FPR",
                "BalancedAcc", "MCC", "n_matched", "n_fake", "n_real", "has_score", "score_flipped"]
        first = not args.out_csv.exists()
        with open(args.out_csv, "a", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            if first:
                w.writeheader()
            row = {"label": label, **r}
            # Format Nones as empty strings for missing AUC/TPR@1%FPR
            for k in ("AUC", "TPR@1%FPR"):
                if row[k] is None:
                    row[k] = ""
            w.writerow(row)
        print(f"\nAppended row to: {args.out_csv}")


if __name__ == "__main__":
    main()
