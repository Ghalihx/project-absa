import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def _save_confusion_matrix(cm, labels, out_png, title="Confusion Matrix"):
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def compute_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame, out_dir: str):
    required = {"review_id", "aspect", "sentiment"}
    if not required.issubset(set(pred_df.columns)) or not required.issubset(set(gt_df.columns)):
        raise ValueError("Kolom wajib: review_id, aspect, sentiment")

    pred = pred_df.copy()
    gt = gt_df.copy()

    pred["review_id"] = pred["review_id"].astype(int)
    gt["review_id"] = gt["review_id"].astype(int)
    for c in ["aspect","sentiment"]:
        pred[c] = pred[c].astype(str).str.strip()
        gt[c] = gt[c].astype(str).str.strip()
    pred["sentiment"] = pred["sentiment"].str.lower()
    gt["sentiment"] = gt["sentiment"].str.lower()

    merged = gt.merge(pred, on=["review_id","aspect"], how="left", suffixes=("_gt","_pred"))
    merged["sentiment_pred"] = merged["sentiment_pred"].fillna("missing")

    y_true = (merged["aspect"] + "|" + merged["sentiment_gt"]).tolist()
    y_pred = (merged["aspect"] + "|" + merged["sentiment_pred"]).tolist()
    labels_pair = sorted(set(y_true) | set(y_pred))

    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels_pair, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    per_class = [{"label":lab, "precision":round(float(pp),4), "recall":round(float(rr),4),
                  "f1":round(float(ff),4), "support":int(ss)}
                 for lab,pp,rr,ff,ss in zip(labels_pair,p,r,f1,sup)]

    overall = {
        "accuracy": round(float(acc),4),
        "macro_precision": round(float(np.mean(p)),4) if len(p) else 0.0,
        "macro_recall": round(float(np.mean(r)),4) if len(r) else 0.0,
        "macro_f1": round(float(np.mean(f1)),4) if len(f1) else 0.0,
    }

    sent_labels = ["negative","neutral","positive","missing"]
    cm_sent = confusion_matrix(merged["sentiment_gt"], merged["sentiment_pred"], labels=sent_labels)
    os.makedirs(out_dir, exist_ok=True)
    cm_png = os.path.join(out_dir, "confusion_matrix_sentiment.png")
    _save_confusion_matrix(cm_sent, sent_labels, cm_png, title="Confusion Matrix (Sentiment)")

    pair_cm_png = ""
    if len(labels_pair) <= 25:
        cm_pair = confusion_matrix(y_true, y_pred, labels=labels_pair)
        pair_cm_png = os.path.join(out_dir, "confusion_matrix_pair.png")
        _save_confusion_matrix(cm_pair, labels_pair, pair_cm_png, title="Confusion Matrix (Aspect|Sentiment)")

    return {
        "overall": overall,
        "per_class": per_class,
        "rows": int(len(merged)),
        "cm_sentiment_png": os.path.basename(cm_png),
        "cm_pair_png": os.path.basename(pair_cm_png) if pair_cm_png else "",
    }
