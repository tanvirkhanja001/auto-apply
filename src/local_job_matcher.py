import json
from typing import Any, Dict, List, Optional
from urllib import request, error


OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama_model(model_name: str, prompt: str) -> Optional[str]:
    """Call a local Ollama model if it is installed and running."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            response = json.loads(body)
            return response.get("response", "")
    except (error.URLError, error.HTTPError, TimeoutError, ValueError):
        return None


def parse_local_model_json(raw_text: Optional[str]) -> Dict[str, Any]:
    if not raw_text:
        return {}

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def match_candidate_with_local_model(
    jd_text: str,
    candidate: Dict[str, Any],
    questions: Optional[List[str]] = None,
    model_name: str = "llama3.1:8b",
) -> Dict[str, Any]:
    """Ask a local model to rate and explain a job match, returning a JSON object."""
    prompt = f"""
You are a local job-matching assistant.
Candidate Profile:
{json.dumps(candidate, ensure_ascii=False)}

Job Description:
{jd_text}

Screening Questions:
{json.dumps(questions or [], ensure_ascii=False)}

Return STRICT JSON with structure:
{
  "match_score": 0,
  "should_apply": false,
  "reason": "short reason",
  "answers": [
    {"question": "question text", "answer": "short answer"}
  ]
}

Rules:
1. Use score 0-100.
2. should_apply is true only if score >= 70.
3. Keep reasoning brief and factual.
4. Output ONLY valid JSON.
"""

    raw = call_ollama_model(model_name, prompt)
    result = parse_local_model_json(raw)
    if not result:
        return {}

    return {
        "match_score": int(result.get("match_score", 0)),
        "should_apply": bool(result.get("should_apply", False)),
        "reason": str(result.get("reason", "Local model did not provide a reason.")),
        "answers": result.get("answers", []),
        "model_name": model_name,
    }
