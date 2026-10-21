import json
import re
from typing import List, Dict, Any

from .llm_client import chat_completion

SENTIMENTS = {"positive", "neutral", "negative"}

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

DEMO_RULES = {
    "cleanliness": {
        "any": ["bersih", "kotor", "sampah", "dirty", "trash"],
        "pos": ["bersih", "clean"],
        "neg": ["kotor", "sampah", "dirty", "trash"],
    },
    "environment/sea quality": {
        "any": ["jernih", "keruh", "air", "waves", "water", "coral", "karang", "ombak"],
        "pos": ["jernih", "clear", "great"],
        "neg": ["keruh", "murky"],
    },
    "accessibility/parking": {
        "any": ["akses", "jalan", "parkir", "parking", "transport"],
        "pos": ["mudah", "easy", "near", "dekat"],
        "neg": ["susah", "macet", "sempit", "far", "difficult", "rusak"],
    },
    "facilities": {
        "any": ["toilet", "restroom", "fasilitas", "wc", "kamar mandi"],
        "pos": ["bagus", "good", "clean"],
        "neg": ["rusak", "kotor", "dirty", "broken", "bau"],
    },
    "price/value": {
        "any": ["mahal", "murah", "harga", "tiket", "price", "expensive", "cheap", "worth"],
        "pos": ["murah", "cheap", "worth", "terjangkau"],
        "neg": ["mahal", "expensive"],
    },
    "service": {
        "any": ["petugas", "staff", "pelayanan", "service", "ramah", "friendly", "rude"],
        "pos": ["ramah", "friendly", "helpful", "cepat"],
        "neg": ["jutek", "rude", "lambat", "slow"],
    },
    "safety": {
        "any": ["aman", "bahaya", "licin", "safe", "danger"],
        "pos": ["aman", "safe"],
        "neg": ["bahaya", "licin", "danger"],
    },
    "crowding": {
        "any": ["ramai", "penuh", "crowded", "queue", "antri"],
        "pos": ["sepi", "quiet"],
        "neg": ["ramai", "penuh", "crowded", "queue", "antri"],
    },
}

def _parse_json(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def build_messages(text: str, scheme: str) -> List[Dict[str, str]]:
    system = (
        "You are an aspect-based sentiment analysis (ABSA) engine for marine tourism reviews. "
        "Extract aspects and assign sentiment polarity per aspect (positive/neutral/negative). "
        "Provide an evidence phrase copied verbatim from the text. "
        "Return STRICT JSON with key 'aspects' as an array of objects: "
        "{aspect, sentiment, evidence, confidence, explanation}. "
        "Explanation: brief reason why the sentiment is assigned, based on evidence."
    )

    if scheme == "few":
        user = FEW_SHOT_EXAMPLES + f"\nText: {text}\nOutput:"
    else:
        user = (
            f"Aspects to consider: {', '.join(ASPECTS)} (or 'other').\n"
            "Rules: evidence must be a direct substring from the text. Prefer 1-6 aspects.\n"
            "Output JSON schema: {\"aspects\":[{\"aspect\":\"...\",\"sentiment\":\"positive|neutral|negative\","
            "\"evidence\":\"...\",\"confidence\":0.0,\"explanation\":\"...\"}]}\n\n"
            f"Text: {text}"
        )
    return [{"role":"system","content":system},{"role":"user","content":user}]

def demo_absa(text: str) -> List[Dict[str, Any]]:
    t = (text or "")
    low = t.lower()
    out = []
    for asp, rule in DEMO_RULES.items():
        if any(k in low for k in rule["any"]):
            sentiment = "neutral"
            matched = None
            for k in rule["pos"]:
                if k in low:
                    sentiment = "positive"; matched = k; break
            for k in rule["neg"]:
                if k in low:
                    sentiment = "negative"; matched = k; break
            if not matched:
                matched = next((k for k in rule["any"] if k in low), "")
            evidence = matched
            explanation = f"Keyword match '{matched}' -> {sentiment} ({asp})."
            out.append({"aspect": asp, "sentiment": sentiment, "evidence": evidence, "confidence": 0.55, "explanation": explanation})
    return out[:8] or [{"aspect":"other","sentiment":"neutral","evidence":"","confidence":0.50,"explanation":"No clear aspect keywords."}]

def run_absa_one(text: str, mode: str="demo", scheme: str="zero") -> List[Dict[str, Any]]:
    if mode == "demo":
        return demo_absa(text)

    msgs = build_messages(text, scheme="few" if scheme=="few" else "zero")
    raw = chat_completion(msgs)
    data = _parse_json(raw)
    aspects = data.get("aspects", []) if isinstance(data, dict) else []
    clean = []
    for a in aspects:
        if not isinstance(a, dict):
            continue
        aspect = str(a.get("aspect","other")).strip()
        sentiment = str(a.get("sentiment","neutral")).strip().lower()
        evidence = str(a.get("evidence","")).strip()
        explanation = str(a.get("explanation","")).strip()
        conf = a.get("confidence", 0.6)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.6
        if sentiment not in SENTIMENTS:
            sentiment = "neutral"
        clean.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "evidence": evidence,
            "confidence": max(0.0, min(1.0, conf)),
            "explanation": explanation
        })
    return clean[:10] or [{"aspect":"other","sentiment":"neutral","evidence":"","confidence":0.5,"explanation":"No aspects returned."}]
