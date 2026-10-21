from __future__ import annotations

import os
import re
import uuid
import json
from typing import Dict, Any, List, Optional

import pandas as pd
import ftfy
import regex as reg
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

from utils.absa import run_absa_one
from utils.metrics import compute_metrics
from utils.plots import plot_keyword_contributions, plot_confusion_matrix

load_dotenv()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
OUTPUT_DIR = os.path.join(APP_ROOT, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
EMOJI_RE = reg.compile(r"\p{Emoji_Presentation}|\p{Emoji}\uFE0F", flags=reg.UNICODE)
PUNCT_KEEP_RE = re.compile(r"[^0-9a-zA-Z\s\.\,\?\!\#\-\']+")

SLANG_MAP_ID = {
    "bgt": "banget", "bgtt": "banget",
    "gk": "gak", "ga": "gak", "nggak": "gak",
    "udh": "sudah", "sdh": "sudah",
    "yg": "yang", "krn": "karena",
    "aja": "saja", "trs": "terus",
    "bgs": "bagus",
}

LIMITED_STOPWORDS_ID = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "saja", "nya"}
LIMITED_STOPWORDS_EN = {"the", "and", "to", "of", "in", "is", "it", "this", "that"}

MARINE_KWS = [
    "pantai", "laut", "snorkel", "diving", "pulau", "karang", "ombak", "sunset",
    "beach", "ocean", "sea", "wave", "island", "coral", "air", "jernih"
]

TEXT_COL_CANDIDATES = [
    "text", "review", "reviews", "comment", "comments", "caption", "content", "message",
    "review_text", "comment_text", "body", "description"
]

ASPECTS = [
    "environment/sea quality", "cleanliness", "facilities", "service",
    "accessibility/parking", "safety", "price/value", "crowding"
]

FEW_SHOT_EXAMPLES = """Example 1:
Text: pantainya bersih tapi parkir susah.
Output: {"aspects":[
 {"aspect":"cleanliness","sentiment":"positive","evidence":"bersih","confidence":0.75,"explanation":"Evidence shows cleanliness is good."},
 {"aspect":"accessibility/parking","sentiment":"negative","evidence":"parkir susah","confidence":0.78,"explanation":"Parking is difficult."}
]}

Example 2:
Text: great waves and clear water, but the restroom is dirty.
Output: {"aspects":[
 {"aspect":"environment/sea quality","sentiment":"positive","evidence":"clear water","confidence":0.80,"explanation":"Clear water indicates good sea quality."},
 {"aspect":"facilities","sentiment":"negative","evidence":"restroom is dirty","confidence":0.82,"explanation":"Dirty restroom indicates poor facilities."}
]}
"""


def read_csv_robust(path: str) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def pick_text_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    hits: List[str] = []
    for key in TEXT_COL_CANDIDATES:
        for c in cols:
            if key == c.lower() or key in c.lower():
                hits.append(c)
    hits = list(dict.fromkeys(hits))
    if hits:
        return hits

    best = None
    best_len = -1.0
    for c in cols:
        if df[c].dtype == "object":
            s = df[c].dropna().astype(str).head(200)
            if len(s) == 0:
                continue
            m = s.str.len().mean()
            if m > best_len:
                best_len = float(m)
                best = c
    return [best] if best else []


def normalize_text(s: str, remove_url: bool=True, remove_html: bool=True, remove_emoji: bool=True) -> str:
    s = "" if s is None else str(s)
    s = ftfy.fix_text(s)
    if remove_html:
        s = HTML_RE.sub(" ", s)
    if remove_url:
        s = URL_RE.sub(" ", s)
    if remove_emoji:
        s = EMOJI_RE.sub(" ", s)
    s = s.replace("\u200b", " ")
    s = s.replace("#", " #").replace("@", " @")
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[’]", "'", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def finalize_text(text: str) -> str:
    text = PUNCT_KEEP_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def case_folding(text: str) -> str:
    return (text or "").lower()


def normalize_slang_id(text: str) -> str:
    toks = (text or "").split()
    toks = [SLANG_MAP_ID.get(t, t) for t in toks]
    return " ".join(toks)


def detect_lang_safe(text: str) -> str:
    t = (text or "").strip()
    if len(t) < 10:
        return "unknown"
    low = t.lower()
    if any(w in low.split() for w in ["yang","dan","di","ke","dari","pantai","laut"]):
        return "id"
    if any(w in low.split() for w in ["the","and","to","of","beach","ocean"]):
        return "en"
    return "unknown"


def code_mixed_heuristic(text: str, lang: str) -> str:
    t = (text or "").lower()
    toks = set(re.findall(r"[a-zA-Z]+", t))
    has_id = len(toks.intersection({"pantai", "laut", "sampah", "akses", "ramai"})) > 0
    has_en = len(toks.intersection({"beach", "ocean", "waves", "parking", "restroom"})) > 0
    if lang in {"id", "en"} and has_id and has_en:
        return "code-mixed"
    return lang


def limited_stopword_removal(text: str, lang: str) -> str:
    toks = (text or "").split()
    if lang in {"id", "code-mixed"}:
        toks = [t for t in toks if t not in LIMITED_STOPWORDS_ID]
    elif lang == "en":
        toks = [t for t in toks if t not in LIMITED_STOPWORDS_EN]
    return " ".join(toks)


def is_marine_relevant(text: str) -> bool:
    t = (text or "").lower()
    for kw in MARINE_KWS:
        if kw in t or f"#{kw}" in t:
            return True
    return False


def build_zero_shot_prompt(text: str) -> str:
    return (
        "You are an expert in aspect-based sentiment analysis for marine tourism.\n"
        "Task: Extract aspects and classify sentiment for each aspect (positive/negative/neutral).\n"
        f"Aspects to consider: {', '.join(ASPECTS)}.\n"
        "Also provide evidence phrases from the text for each aspect.\n"
        "Return JSON array with fields: aspect, sentiment, evidence.\n\n"
        f"Text: {text}"
    )


def build_few_shot_prompt(text: str) -> str:
    return (
        "You are an expert in aspect-based sentiment analysis for marine tourism.\n"
        "Follow the examples and produce the same JSON format.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        f"Text: {text}\n"
        "Output:"
    )


def preprocess_text(raw: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    t = normalize_text(raw, opts["remove_url"], opts["remove_html"], opts["remove_emoji"])
    if opts["case_folding"]:
        t = case_folding(t)
    if opts["slang_norm"]:
        t = normalize_slang_id(t)
    t = finalize_text(t)

    lang = detect_lang_safe(t)
    if opts["code_mixed"]:
        lang = code_mixed_heuristic(t, lang)

    t_sw = t
    if opts["limited_stopwords"]:
        t_sw = limited_stopword_removal(t, lang)
        t_sw = finalize_text(t_sw)

    relevant = True
    if opts["relevance_filter"]:
        relevant = is_marine_relevant(t_sw)

    # Prompts for quick testing (shown in preprocess preview)
    p_zero = build_zero_shot_prompt(t_sw)
    p_few = build_few_shot_prompt(t_sw)

    return {
        "text_raw": raw,
        "text_clean": t,
        "text_clean_sw": t_sw,
        "language": lang,
        "text_length": len(t_sw),
        "is_relevant": relevant,
        "prompt_zero_shot": p_zero,
        "prompt_few_shot": p_few,
    }


def default_options() -> Dict[str, bool]:
    return {
        "remove_url": True,
        "remove_html": True,
        "remove_emoji": True,
        "case_folding": True,
        "slang_norm": True,
        "code_mixed": True,
        "limited_stopwords": True,
        "relevance_filter": True,
    }


def get_options_from_form(form) -> Dict[str, bool]:
    opts = default_options()
    for k in list(opts.keys()):
        opts[k] = (form.get(k) == "on")
    return opts


def job_paths(job_id: str) -> Dict[str, str]:
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    pre_csv = os.path.join(out_dir, f"{job_id}.preprocessed.csv")

    pred_zero = os.path.join(out_dir, f"{job_id}.predictions_zero.csv")
    pred_few = os.path.join(out_dir, f"{job_id}.predictions_few.csv")

    xai_zero = os.path.join(out_dir, f"{job_id}.xai_keywords_zero.png")
    xai_few = os.path.join(out_dir, f"{job_id}.xai_keywords_few.png")

    cm_zero = os.path.join(out_dir, f"{job_id}.confusion_zero.png")
    cm_few = os.path.join(out_dir, f"{job_id}.confusion_few.png")

    meta_path = os.path.join(out_dir, f"{job_id}.meta.json")
    return {
        "dir": out_dir,
        "pre_csv": pre_csv,
        "pred_zero": pred_zero,
        "pred_few": pred_few,
        "xai_zero": xai_zero,
        "xai_few": xai_few,
        "cm_zero": cm_zero,
        "cm_few": cm_few,
        "meta": meta_path,
    }


def find_upload_by_id(file_id: str) -> Optional[str]:
    for fn in os.listdir(UPLOAD_DIR):
        if fn.startswith(file_id + "__"):
            return os.path.join(UPLOAD_DIR, fn)
    return None


def _read_meta(paths: Dict[str, str]) -> Dict[str, Any]:
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_meta(paths: Dict[str, str], meta: Dict[str, Any]) -> None:
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@app.get("/")
def home():
    return render_template(
        "index.html",
        uploaded=False,
        file_id="",
        filename="",
        columns=[],
        inferred_cols=[],
        preview_rows=[],
        processed_preview=[],
        opts=default_options(),
        selected_col="",
        job_id="",
        message=""
    )


@app.post("/upload")
def upload():
    f = request.files.get("csv_file")
    if not f or f.filename.strip() == "":
        flash("Please choose a CSV file to upload.", "error")
        return redirect(url_for("home"))

    file_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^a-zA-Z0-9\.\-\_]+", "_", f.filename)
    saved_path = os.path.join(UPLOAD_DIR, f"{file_id}__{safe_name}")
    f.save(saved_path)

    try:
        df = read_csv_robust(saved_path)
    except Exception as e:
        flash(f"Failed to read CSV: {e}", "error")
        return redirect(url_for("home"))

    cols = list(df.columns)
    inferred = pick_text_columns(df)
    preview = df.head(10).fillna("").to_dict(orient="records")

    return render_template(
        "index.html",
        uploaded=True,
        file_id=file_id,
        filename=safe_name,
        columns=cols,
        inferred_cols=inferred,
        preview_rows=preview,
        processed_preview=[],
        opts=default_options(),
        selected_col=(inferred[0] if inferred else (cols[0] if cols else "")),
        job_id="",
        message="File uploaded. Select a text column and run preprocessing."
    )


@app.post("/process")
def process():
    file_id = request.form.get("file_id", "").strip()
    filename = request.form.get("filename", "").strip()
    selected_col = request.form.get("text_col", "").strip()

    if not file_id:
        flash("Missing file_id. Please upload a CSV first.", "error")
        return redirect(url_for("home"))

    upload_path = find_upload_by_id(file_id)
    if not upload_path or not os.path.exists(upload_path):
        flash("Uploaded file not found. Please upload again.", "error")
        return redirect(url_for("home"))

    df = read_csv_robust(upload_path)
    cols = list(df.columns)
    inferred = pick_text_columns(df)
    preview = df.head(10).fillna("").to_dict(orient="records")

    if selected_col == "" or selected_col not in df.columns:
        selected_col = inferred[0] if inferred else (cols[0] if cols else "")
    if selected_col == "":
        flash("No valid text column found.", "error")
        return redirect(url_for("home"))

    opts = get_options_from_form(request.form)

    raw_series = df[selected_col].fillna("").astype(str)
    processed = [preprocess_text(raw, opts) for raw in raw_series.tolist()]

    proc_df = df.copy()
    proc_df["text_raw"] = [p["text_raw"] for p in processed]
    proc_df["text_clean"] = [p["text_clean"] for p in processed]
    proc_df["text_clean_sw"] = [p["text_clean_sw"] for p in processed]
    proc_df["language"] = [p["language"] for p in processed]
    proc_df["text_length"] = [p["text_length"] for p in processed]
    proc_df["is_relevant"] = [p["is_relevant"] for p in processed]
    proc_df["prompt_zero_shot"] = [p["prompt_zero_shot"] for p in processed]
    proc_df["prompt_few_shot"] = [p["prompt_few_shot"] for p in processed]

    if opts["relevance_filter"]:
        proc_df = proc_df[proc_df["is_relevant"] == True].copy()

    job_id = str(uuid.uuid4())
    paths = job_paths(job_id)
    proc_df.to_csv(paths["pre_csv"], index=False)

    meta = {
        "job_id": job_id,
        "file_id": file_id,
        "filename": filename,
        "selected_col": selected_col,
        "options": opts,
        "rows_in": int(len(df)),
        "rows_out": int(len(proc_df)),
        "status": "preprocessed",
        "last_scheme": ""
    }
    _write_meta(paths, meta)

    processed_preview = proc_df.head(10).fillna("").to_dict(orient="records")

    return render_template(
        "index.html",
        uploaded=True,
        file_id=file_id,
        filename=filename,
        columns=cols,
        inferred_cols=inferred,
        preview_rows=preview,
        processed_preview=processed_preview,
        opts=opts,
        selected_col=selected_col,
        job_id=job_id,
        message=f"Preprocessing finished. Rows out: {meta['rows_out']} (from {meta['rows_in']})."
    )


@app.post("/analyze")
def analyze():
    job_id = request.form.get("job_id", "").strip()
    mode = request.form.get("mode", "demo").strip()
    scheme = request.form.get("scheme", "zero").strip()
    scheme = "few" if scheme == "few" else "zero"

    if not job_id:
        flash("job_id kosong. Jalankan preprocessing dulu.", "error")
        return redirect(url_for("home"))

    paths = job_paths(job_id)
    if not os.path.exists(paths["pre_csv"]):
        flash("Preprocessed CSV tidak ditemukan. Jalankan preprocessing dulu.", "error")
        return redirect(url_for("home"))

    df = pd.read_csv(paths["pre_csv"])
    if "text_clean_sw" not in df.columns:
        flash("Kolom text_clean_sw tidak ditemukan.", "error")
        return redirect(url_for("home"))

    rows = []
    for i, txt in enumerate(df["text_clean_sw"].fillna("").astype(str).tolist(), start=1):
        aspects = run_absa_one(txt, mode=mode, scheme=("few" if scheme=="few" else "zero"))
        for a in aspects:
            rows.append({
                "review_id": i,
                "aspect": a.get("aspect"),
                "sentiment": a.get("sentiment"),
                "evidence": a.get("evidence"),
                "confidence": a.get("confidence"),
                "explanation": a.get("explanation"),
                "text": txt
            })

    pred_df = pd.DataFrame(rows)

    pred_path = paths["pred_few"] if scheme == "few" else paths["pred_zero"]
    pred_df.to_csv(pred_path, index=False)

    xai_path = paths["xai_few"] if scheme == "few" else paths["xai_zero"]
    try:
        plot_keyword_contributions(pred_df[["sentiment","text"]], xai_path, top_k=10)
    except Exception as e:
        flash(f"XAI plot gagal dibuat: {e}", "error")

    meta = _read_meta(paths)
    meta.update({
        "status": "analyzed",
        "absa_mode": mode,
        "last_scheme": scheme,
        "pred_rows_"+scheme: int(len(pred_df))
    })
    _write_meta(paths, meta)

    return redirect(url_for("results", job_id=job_id, scheme=scheme))


@app.get("/results/<job_id>")
def results(job_id: str):
    scheme = request.args.get("scheme", "").strip().lower()
    paths = job_paths(job_id)
    meta = _read_meta(paths)
    if scheme not in {"zero","few"}:
        scheme = meta.get("last_scheme") or "zero"

    pred_path = paths["pred_few"] if scheme == "few" else paths["pred_zero"]
    if not os.path.exists(pred_path):
        flash("Predictions belum ada untuk skema ini. Klik Analyze dulu.", "error")
        return redirect(url_for("home"))

    df = pd.read_csv(pred_path).fillna("")
    summary = []
    if len(df):
        summary = (df.groupby(["aspect","sentiment"]).size().reset_index(name="count")
                     .sort_values(["aspect","count"], ascending=[True, False]).to_dict(orient="records"))
    preview = df.head(200).to_dict(orient="records")

    xai_path = paths["xai_few"] if scheme == "few" else paths["xai_zero"]
    xai_url = url_for("job_asset", job_id=job_id, filename=os.path.basename(xai_path)) if os.path.exists(xai_path) else ""

    cm_path = paths["cm_few"] if scheme == "few" else paths["cm_zero"]
    cm_url = url_for("job_asset", job_id=job_id, filename=os.path.basename(cm_path)) if os.path.exists(cm_path) else ""

    return render_template(
        "results.html",
        job_id=job_id,
        meta=meta,
        summary=summary,
        rows=preview,
        xai_url=xai_url,
        cm_url=cm_url,
        scheme=scheme
    )


@app.post("/make_confusion/<job_id>")
def make_confusion(job_id: str):
    # Confusion matrix 3×3 needs ground_truth_overall.csv: review_id, overall
    scheme = request.form.get("scheme", "zero").strip().lower()
    scheme = "few" if scheme == "few" else "zero"
    gt_file = request.files.get("gt_overall")

    if not gt_file or not gt_file.filename:
        flash("Upload ground truth overall CSV dulu (kolom: review_id, overall).", "error")
        return redirect(url_for("results", job_id=job_id, scheme=scheme))

    paths = job_paths(job_id)
    pred_path = paths["pred_few"] if scheme == "few" else paths["pred_zero"]
    if not os.path.exists(pred_path):
        flash("Predictions belum ada untuk skema ini.", "error")
        return redirect(url_for("results", job_id=job_id, scheme=scheme))

    try:
        gt = pd.read_csv(gt_file)
    except Exception as e:
        flash(f"Gagal baca ground truth: {e}", "error")
        return redirect(url_for("results", job_id=job_id, scheme=scheme))

    if not {"review_id","overall"}.issubset(set(gt.columns)):
        flash("Ground truth overall harus punya kolom: review_id, overall.", "error")
        return redirect(url_for("results", job_id=job_id, scheme=scheme))

    gt2 = gt.copy()
    gt2["review_id"] = gt2["review_id"].astype(int)
    gt2["overall"] = gt2["overall"].astype(str).str.title()

    pred = pd.read_csv(pred_path)
    pred["sentiment"] = pred["sentiment"].astype(str).str.lower()
    score = {"negative": -1, "neutral": 0, "positive": 1}
    pred["s"] = pred["sentiment"].map(score).fillna(0)
    g = pred.groupby("review_id")["s"].mean().reset_index()

    def to_label(v):
        if v > 0.15: return "Positive"
        if v < -0.15: return "Negative"
        return "Neutral"
    g["overall_pred"] = g["s"].apply(to_label)

    m = gt2.merge(g[["review_id","overall_pred"]], on="review_id", how="left")
    m["overall_pred"] = m["overall_pred"].fillna("Neutral")

    labels = ["Negative","Neutral","Positive"]
    cm_path = paths["cm_few"] if scheme == "few" else paths["cm_zero"]
    plot_confusion_matrix(m["overall"], m["overall_pred"], labels, cm_path,
                          f"Confusion Matrix: {'Few-Shot' if scheme=='few' else 'Zero-Shot'}")

    flash("Confusion matrix berhasil dibuat.", "success")
    return redirect(url_for("results", job_id=job_id, scheme=scheme))


@app.get("/job_asset/<job_id>/<filename>")
def job_asset(job_id: str, filename: str):
    paths = job_paths(job_id)
    target = os.path.join(paths["dir"], filename)
    if not os.path.exists(target):
        return {"error":"file not found"}, 404
    return send_file(target, as_attachment=False)


@app.get("/download/<job_id>.preprocessed.csv")
def download_preprocessed(job_id: str):
    paths = job_paths(job_id)
    if not os.path.exists(paths["pre_csv"]):
        flash("Output CSV tidak ditemukan.", "error")
        return redirect(url_for("home"))
    return send_file(paths["pre_csv"], as_attachment=True, download_name=f"preprocessed_{job_id}.csv")


@app.get("/download/<job_id>.predictions_<scheme>.csv")
def download_predictions(job_id: str, scheme: str):
    scheme = (scheme or "").strip().lower()
    scheme = "few" if scheme == "few" else "zero"
    paths = job_paths(job_id)
    pred_path = paths["pred_few"] if scheme == "few" else paths["pred_zero"]
    if not os.path.exists(pred_path):
        flash("Predictions CSV belum ada.", "error")
        return redirect(url_for("home"))
    return send_file(pred_path, as_attachment=True, download_name=f"predictions_{scheme}_{job_id}.csv")


@app.get("/meta/<job_id>")
def view_meta(job_id: str):
    paths = job_paths(job_id)
    if not os.path.exists(paths["meta"]):
        return {"error": "meta not found"}, 404
    with open(paths["meta"], "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/evaluate")
def evaluate_get():
    return render_template("evaluate.html")


@app.post("/evaluate")
def evaluate_post():
    pred_file = request.files.get("pred_file")
    gt_file = request.files.get("gt_file")
    if not pred_file or not gt_file or not pred_file.filename or not gt_file.filename:
        flash("Upload file prediksi dan ground truth.", "error")
        return redirect(url_for("evaluate_get"))

    pred = pd.read_csv(pred_file)
    gt = pd.read_csv(gt_file)

    eval_id = f"eval_{uuid.uuid4()}"
    out_dir = os.path.join(OUTPUT_DIR, eval_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        report = compute_metrics(pred, gt, out_dir=out_dir)
    except Exception as e:
        flash(f"Evaluasi gagal: {e}", "error")
        return redirect(url_for("evaluate_get"))

    report["cm_sentiment_url"] = url_for("eval_asset", eval_id=eval_id, filename=report["cm_sentiment_png"])
    report["cm_pair_url"] = url_for("eval_asset", eval_id=eval_id, filename=report["cm_pair_png"]) if report.get("cm_pair_png") else ""
    return render_template("evaluate_result.html", report=report)


@app.get("/evaluate_compare")
def evaluate_compare_get():
    return render_template("evaluate_compare.html")


@app.post("/evaluate_compare")
def evaluate_compare_post():
    pred_zero = request.files.get("pred_zero")
    pred_few = request.files.get("pred_few")
    gt_file = request.files.get("gt_file")

    if not pred_zero or not pred_few or not gt_file:
        flash("Upload predictions zero-shot, predictions few-shot, dan ground truth.", "error")
        return redirect(url_for("evaluate_compare_get"))

    df_zero = pd.read_csv(pred_zero)
    df_few = pd.read_csv(pred_few)
    gt = pd.read_csv(gt_file)

    def agg_overall(df):
        d = df.copy()
        d["sentiment"] = d["sentiment"].astype(str).str.lower()
        score = {"negative": -1, "neutral": 0, "positive": 1}
        d["s"] = d["sentiment"].map(score).fillna(0)
        g = d.groupby("review_id")["s"].mean().reset_index()
        def to_label(v):
            if v > 0.15: return "Positive"
            if v < -0.15: return "Negative"
            return "Neutral"
        g["overall_pred"] = g["s"].apply(to_label)
        return g[["review_id","overall_pred"]]

    if not {"review_id","overall"}.issubset(set(gt.columns)):
        flash("Ground truth harus punya kolom: review_id, overall (Negative/Neutral/Positive).", "error")
        return redirect(url_for("evaluate_compare_get"))

    gt2 = gt.copy()
    gt2["review_id"] = gt2["review_id"].astype(int)
    gt2["overall"] = gt2["overall"].astype(str).str.title()

    z = agg_overall(df_zero)
    f = agg_overall(df_few)

    m0 = gt2.merge(z, on="review_id", how="left")
    m1 = gt2.merge(f, on="review_id", how="left")
    m0["overall_pred"] = m0["overall_pred"].fillna("Neutral")
    m1["overall_pred"] = m1["overall_pred"].fillna("Neutral")

    labels = ["Negative","Neutral","Positive"]

    eval_id = f"evalcmp_{uuid.uuid4()}"
    out_dir = os.path.join(OUTPUT_DIR, eval_id)
    os.makedirs(out_dir, exist_ok=True)

    plot_confusion_matrix(m0["overall"], m0["overall_pred"], labels, os.path.join(out_dir, "confusion_zero.png"), "Confusion Matrix: Zero-Shot")
    plot_confusion_matrix(m1["overall"], m1["overall_pred"], labels, os.path.join(out_dir, "confusion_few.png"), "Confusion Matrix: Few-Shot")

    try:
        plot_keyword_contributions(df_zero[["sentiment","text"]], os.path.join(out_dir, "xai_keywords_zero.png"), top_k=10)
        plot_keyword_contributions(df_few[["sentiment","text"]], os.path.join(out_dir, "xai_keywords_few.png"), top_k=10)
    except Exception as e:
        flash(f"XAI keyword plot compare gagal: {e}", "error")

    return render_template(
        "evaluate_compare_result.html",
        cm_zero_url=url_for("eval_asset", eval_id=eval_id, filename="confusion_zero.png"),
        cm_few_url=url_for("eval_asset", eval_id=eval_id, filename="confusion_few.png"),
        xai_zero_url=url_for("eval_asset", eval_id=eval_id, filename="xai_keywords_zero.png"),
        xai_few_url=url_for("eval_asset", eval_id=eval_id, filename="xai_keywords_few.png"),
    )


@app.get("/eval_asset/<eval_id>/<filename>")
def eval_asset(eval_id: str, filename: str):
    out_dir = os.path.join(OUTPUT_DIR, eval_id)
    target = os.path.join(out_dir, filename)
    if not os.path.exists(target):
        return {"error":"file not found"}, 404
    return send_file(target, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)
