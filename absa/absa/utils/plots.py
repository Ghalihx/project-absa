import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix

WORD_RE = re.compile(r"[A-Za-z']{2,}")

def _tokenize(s: str):
    return WORD_RE.findall((s or "").lower())

def _log_odds_ratio(counts_a: Counter, counts_b: Counter, vocab: list, alpha: float=0.01):
    """Smoothed log-odds ratio (Monroe et al. style, simplified)."""
    a_total = sum(counts_a.values()) + alpha * len(vocab)
    b_total = sum(counts_b.values()) + alpha * len(vocab)
    weights = {}
    for w in vocab:
        a = counts_a.get(w, 0) + alpha
        b = counts_b.get(w, 0) + alpha
        weights[w] = np.log(a / (a_total - a)) - np.log(b / (b_total - b))
    return weights

def plot_keyword_contributions(pred_df: pd.DataFrame, out_png: str, top_k: int=8):
    """3-panel keyword contribution plot (Positive/Neutral/Negative).

    Fix: sebelumnya hanya ambil top positif => semua hijau.
    Sekarang ambil top positif + top negatif agar bar merah juga muncul (seperti contoh Anda).
    """
    if "sentiment" not in pred_df.columns or "text" not in pred_df.columns:
        raise ValueError("pred_df harus punya kolom: sentiment, text")

    df = pred_df.copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df["text"] = df["text"].astype(str)

    sentiments = ["positive", "neutral", "negative"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Explainable AI: Key Word Contributions Across All Sentiments")

    for ax, s in zip(axes, sentiments):
        texts_s = df.loc[df["sentiment"] == s, "text"].tolist()
        texts_o = df.loc[df["sentiment"] != s, "text"].tolist()

        cnt_s = Counter()
        for t in texts_s:
            cnt_s.update(_tokenize(t))
        cnt_o = Counter()
        for t in texts_o:
            cnt_o.update(_tokenize(t))

        vocab = sorted(set(cnt_s.keys()) | set(cnt_o.keys()))
        ax.set_title(f"XAI: {s.title()} Prediction Keywords")
        ax.set_xlabel("Contribution Weight")
        ax.grid(True, axis="x", alpha=0.3)

        if not vocab or (sum(cnt_s.values()) == 0 and sum(cnt_o.values()) == 0):
            ax.text(0.5, 0.5, "No tokens", ha="center", va="center")
            continue

        weights = _log_odds_ratio(cnt_s, cnt_o, vocab=vocab, alpha=0.01)

        # top positive + top negative (by magnitude)
        items = list(weights.items())
        pos = sorted([it for it in items if it[1] > 0], key=lambda x: x[1], reverse=True)[:max(1, top_k//2)]
        neg = sorted([it for it in items if it[1] < 0], key=lambda x: x[1])[:max(1, top_k//2)]
        top = pos + neg

        # If one side is empty, fallback to absolute top_k
        if len(top) < 3:
            top = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        # Sort for nicer horizontal bar ordering
        top = sorted(top, key=lambda x: x[1])
        words = [w for w,_ in top]
        vals = [v for _,v in top]
        colors = ["#1a7f37" if v >= 0 else "#d1242f" for v in vals]  # strong green/red

        ax.barh(words, vals, color=colors, edgecolor="black", linewidth=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, labels, out_png: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return cm
