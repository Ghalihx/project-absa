import os, json, requests

def chat_completion(messages, temperature=0.2, max_tokens=900):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY belum di-set. Jalankan mode=demo atau set API key.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    timeout = int(os.environ.get("OPENAI_TIMEOUT", "60"))

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM error {r.status_code}: {r.text[:600]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]
