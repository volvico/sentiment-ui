import os, requests
from time import time
from typing import Any, Dict

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def _with_retries(method, url, *, json=None, timeout=30, max_attempts=3, backoff=0.8):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.request(method, url, json=json, timeout=timeout)
            return r
        except requests.RequestException as e:
            last_err = e
            if attempt == max_attempts:
                raise
            time.sleep(backoff * attempt)
    raise last_err  # should not reach

def get_health() -> Dict[str, Any]:
    url = f"{API_BASE_URL}/health"
    r = _with_retries("GET", url, timeout=10)
    r.raise_for_status()
    return r.json()

def predict_one(text: str) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/predict_one"
    r = _with_retries("POST", url, json={"text": text}, timeout=30)
    return {"status_code": r.status_code, "json": (r.json() if r.headers.get("content-type","").startswith("application/json") else None), "text": r.text}

def explain_lime(text: str) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/explain_lime"
    r = _with_retries("POST", url, json={"text": text}, timeout=120)
    return {"status_code": r.status_code, "json": (r.json() if r.headers.get("content-type","").startswith("application/json") else None), "text": r.text}

def predict(texts):
    r = requests.post(f"{API_BASE_URL}/predict", json={"texts": texts}, timeout=30)
    r.raise_for_status()
    return r.json()

def explain(text):
    r = requests.post(f"{API_BASE_URL}/explain", json={"text": text}, timeout=30)
    r.raise_for_status()
    return r.json()
