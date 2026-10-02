import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(".env.local")

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


class ReasoningAgent:
    """Agent-based reasoning with Gemini-first priority and Ollama fallbacks."""

    def __init__(self, model_name: str = "qwen2.5:0.5b-instruct", ollama_url: str = "http://localhost:11434", ollama_model_order: List[str] | None = None):
        self.model_name = model_name
        self.ollama_url = ollama_url
        configured_priority = list(ollama_model_order or [])
        if not configured_priority:
            configured_priority = [
                "gemini-3.5-flash",
                model_name,
                "SmolLM2-360M-Instruct",
            ]
        self.ollama_model_order = [m for m in configured_priority if m and m.strip()]
        # Remove duplicates while preserving order.
        seen = set()
        deduped = []
        for item in self.ollama_model_order:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        self.ollama_model_order = deduped
        if self.model_name not in self.ollama_model_order:
            self.ollama_model_order.insert(0, self.model_name)
        self.remote_available = None
        self.gemini_client = None
        self._response_cache: Dict[str, Any] = {}
        if genai is not None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                except Exception:
                    self.gemini_client = None
        # Quick reachability check for the remote reasoning service
        try:
            from urllib.parse import urlparse
            import socket
            urlp = urlparse(self.ollama_url)
            host = urlp.hostname or "localhost"
            port = urlp.port or (443 if urlp.scheme == "https" else 80)
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            self.remote_available = True
        except Exception:
            self.remote_available = False

    def _cache_key(self, operation: str, *payloads: Any) -> str:
        normalized = json.dumps(payloads, sort_keys=True, default=str)
        return f"{operation}:{normalized}"

    def _get_cached(self, operation: str, *payloads: Any) -> Any:
        key = self._cache_key(operation, *payloads)
        return self._response_cache.get(key)

    def _set_cached(self, operation: str, value: Any, *payloads: Any) -> Any:
        key = self._cache_key(operation, *payloads)
        self._response_cache[key] = value
        return value

    def _call_ollama(self, prompt: str, timeout: int = 3, model_name: str | None = None) -> str:
        if requests is None or not self.remote_available:
            return ""
        target_model = model_name or self.model_name
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=timeout,
            )
            if response.ok:
                answer = response.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception as ex:
            print(f"ReasoningAgent: Ollama model '{target_model}' failed; trying next fallback: {ex}")
        return ""

    def _call_ollama_priority(self, prompt: str) -> str:
        if requests is None or not self.remote_available:
            return ""
        for model_name in self.ollama_model_order:
            answer = self._call_ollama(prompt, model_name=model_name)
            if answer:
                return answer
        return ""

    def _call_gemini(self, prompt: str) -> str:
        if self.gemini_client is None:
            return ""
        try:
            # Google retired several older Gemini model names. Use the currently supported 3.x family.
            models_to_try = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
            last_error = None
            for model_name in models_to_try:
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={
                            "temperature": 0.1,
                            "response_mime_type": "application/json",
                        },
                    )
                    text = getattr(response, "text", "") or ""
                    if text:
                        return text.strip()
                except Exception as ex:
                    last_error = ex
            if last_error:
                print(f"ReasoningAgent: Gemini fallback failed: {last_error}")
        except Exception as ex:
            print(f"ReasoningAgent: Gemini fallback failed: {ex}")
        return ""

    def _build_prompt(self, candidate: Dict[str, Any], top_jobs: List[Dict[str, Any]], fit_score: float) -> str:
        return f"""
You are a local job-fit evaluator.

Candidate profile:
{json.dumps(candidate, indent=2)}

Top matched jobs:
{json.dumps(top_jobs, indent=2)}

Current fit score: {fit_score}

Task:
1. Decide whether the candidate should apply to these jobs.
2. Output only valid JSON with keys:
   - "should_apply": true/false
   - "reason": short explanation
   - "fit_score": integer 0-100
3. Consider skill overlap, years of experience, role alignment, and location fit.
"""

    @staticmethod
    def _normalize_choice_text(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip().lower()) if value else ""

    @staticmethod
    def _normalize_model_text(value: str) -> str:
        text = (value or "").strip()
        if not text:
            return ""

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ["answer", "response", "result", "value", "text"]:
                    candidate_value = parsed.get(key)
                    if isinstance(candidate_value, str) and candidate_value.strip():
                        return candidate_value.strip()
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            inner = fenced.group(1).strip()
            normalized = ReasoningAgent._normalize_model_text(inner)
            if normalized:
                return normalized

        answer_match = re.search(r'"answer"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        return text

    def answer_question(self, candidate: Dict[str, Any], question: str, options: List[str] | None = None) -> str:
        """Return a profile-grounded answer to a screening question and, when options are supplied, pick the closest answer choice."""
        q = (question or "").strip()
        if not q:
            return ""

        cache_key = ("answer_question", candidate, q, options or [])
        cached = self._get_cached(*cache_key)
        if cached is not None:
            return cached

        prompt = f"""
You answer job application screening questions using the candidate profile.
Candidate profile:
{json.dumps(candidate, indent=2)}
Question:
{question}

Return ONLY a short answer, no extra text.
Examples:
- yes
- no
- 30 days
- 18 LPA
- 3 years
- available to join immediately
"""

        for agent_response in [self._call_gemini(prompt), self._call_ollama_priority(prompt)]:
            if agent_response:
                normalized = self._normalize_model_text(agent_response)
                result = self._pick_best_option(candidate, question, normalized, options or [])
                return self._set_cached("answer_question", result, candidate, q, options or [])

        q_lower = q.lower()
        current_location = str(candidate.get("current_location", "") or "")
        preferred_locations = [str(x).strip() for x in candidate.get("preferred_locations", []) if str(x).strip()]
        notice = str(candidate.get("notice_period_days", "30")).strip() or "30"
        expected_ctc = str(candidate.get("expected_ctc_lpa", "14")).strip() or "14"
        current_ctc = str(candidate.get("current_ctc_lpa", "12")).strip() or "12"
        exp = str(candidate.get("total_exp_years", "3")).strip() or "3"

        # Last-resort profile-aware fallback, intentionally narrow and not a heuristic-first path.
        if any(k in q_lower for k in ["current location", "current city", "where do you live", "city are you currently in", "living in", "residing in"]):
            if any(token in q_lower for token in ["relocate", "willing", "open to relocate"]):
                result = self._pick_best_option(candidate, question, "Yes", options or [])
                return self._set_cached("answer_question", result, candidate, q, options or [])
            result = self._pick_best_option(candidate, question, current_location or "Nashik", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(term in q_lower for term in ["relocate", "ready to relocate", "willing to relocate", "open to relocate", "comfortable relocating", "can relocate", "available to relocate", "travel", "moving"]):
            result = self._pick_best_option(candidate, question, "Yes", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(k in q_lower for k in ["notice period", "notice", "available to join", "join in", "join within", "start date", "availability"]):
            result = self._pick_best_option(candidate, question, f"{notice} days", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(k in q_lower for k in ["expected ctc", "expected salary", "salary expectation", "package", "lpa"]):
            result = self._pick_best_option(candidate, question, f"{expected_ctc} LPA", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(k in q_lower for k in ["current ctc", "current salary", "current package"]):
            result = self._pick_best_option(candidate, question, f"{current_ctc} LPA", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(k in q_lower for k in ["experience", "years of experience", "exp", "years of exp", "work experience"]):
            result = self._pick_best_option(candidate, question, f"{exp} years", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        if any(city.lower() in q_lower for city in preferred_locations):
            result = self._pick_best_option(candidate, question, "Yes", options or [])
            return self._set_cached("answer_question", result, candidate, q, options or [])

        result = self._pick_best_option(candidate, question, "Yes, based on my profile and skill fit.", options or [])
        return self._set_cached("answer_question", result, candidate, q, options or [])

    @staticmethod
    def _pick_best_option(candidate: Dict[str, Any], question: str, answer: str, options: List[str]) -> str:
        """Choose the argument's closest option when the question is a radio/select question."""
        if not options:
            return answer

        normalized_answer = (answer or "").strip()
        if not normalized_answer:
            return options[0]

        option_map = []
        for opt in options:
            val = (opt or "").strip()
            if val:
                option_map.append(val)

        # Exact normalized match
        a_norm = ReasoningAgent._normalize_choice_text(normalized_answer)
        for opt in option_map:
            if ReasoningAgent._normalize_choice_text(opt) == a_norm:
                return opt

        # Boolean match from yes/no decision.
        for yes_token in ["yes", "y", "available", "open", "comfortable", "ready", "willing"]:
            if yes_token in a_norm and any(ReasoningAgent._normalize_choice_text(opt) in {"yes", "y", "available", "open", "comfortable", "ready", "willing"} for opt in option_map):
                for opt in option_map:
                    if ReasoningAgent._normalize_choice_text(opt) in {"yes", "y", "available", "open", "comfortable", "ready", "willing"}:
                        return opt
        for no_token in ["no", "n", "not available", "not open", "not willing", "cannot", "unavailable"]:
            if no_token in a_norm and any(ReasoningAgent._normalize_choice_text(opt) in {"no", "n", "not available", "not open", "not willing", "cannot", "unavailable"} for opt in option_map):
                for opt in option_map:
                    if ReasoningAgent._normalize_choice_text(opt) in {"no", "n", "not available", "not open", "not willing", "cannot", "unavailable"}:
                        return opt

        # Numeric / range matching for experience and notice-period options.
        import re
        answer_nums = re.findall(r"\d+", normalized_answer)
        if answer_nums:
            answer_num = int(answer_nums[0])
            for opt in option_map:
                opt_norm = ReasoningAgent._normalize_choice_text(opt)
                # exact numeric match: '3 years' -> '3-5 years' etc
                range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)", opt_norm)
                if range_match:
                    low = int(range_match.group(1)); high = int(range_match.group(2))
                    if low <= answer_num <= high:
                        return opt
                plus_match = re.search(r"(\d+)\s*\+", opt_norm)
                if plus_match and answer_num >= int(plus_match.group(1)):
                    return opt
                lt_match = re.search(r"<\s*(\d+)", opt_norm)
                if lt_match and answer_num < int(lt_match.group(1)):
                    return opt

        # Semantic overlap by token and named skill/location match.
        q_lower = (question or "").lower()
        best_opt = option_map[0]
        best_score = -1
        for opt in option_map:
            opt_norm = ReasoningAgent._normalize_choice_text(opt)
            score = 0
            for token in re.split(r"[^a-z0-9]+", q_lower):
                if token and token in opt_norm:
                    score += 1
            if any(city.lower() in opt_norm for city in ["pune", "bangalore", "mumbai", "nashik", "hyderabad", "chennai", "delhi", "remote"] if city.lower() in q_lower):
                score += 2
            if any(skill.lower() in opt_norm for skill in ["java", "spring", "react", "aws", "docker", "microservices"] if skill.lower() in q_lower):
                score += 2
            if score > best_score:
                best_score = score
                best_opt = opt

        return best_opt

    def match_answer_to_options(self, candidate: Dict[str, Any], question: str, answer: str, options: List[str]) -> str:
        """Match a candidate answer to the best available option using direct agent reasoning first."""
        cache_key = ("match_answer_to_options", candidate, question, answer, options)
        cached = self._get_cached(*cache_key)
        if cached is not None:
            return cached

        if not options:
            return self._set_cached("match_answer_to_options", answer, candidate, question, answer, options)

        prompt = f"""
You are matching a candidate's answer to the best option from a list of radio button/select choices.
Candidate profile:
{json.dumps(candidate, indent=2)}
Question:
{question}
Candidate's answer:
{answer}
Available options:
{json.dumps(options, indent=2)}

Return ONLY the exact option text from the available options that best matches the candidate's answer.
If no option matches well, return the closest match.
"""

        for agent_response in [self._call_gemini(prompt), self._call_ollama_priority(prompt)]:
            if agent_response:
                normalized = self._normalize_model_text(agent_response)
                for opt in options:
                    if opt.lower() in normalized.lower() or normalized.lower() in opt.lower():
                        return self._set_cached("match_answer_to_options", opt, candidate, question, answer, options)
                return self._set_cached("match_answer_to_options", normalized, candidate, question, answer, options)

        try:
            result = self._pick_best_option(candidate, question, answer, options)
            return self._set_cached("match_answer_to_options", result, candidate, question, answer, options)
        except Exception:
            result = options[0]
            return self._set_cached("match_answer_to_options", result, candidate, question, answer, options)

    def build_answers_for_questions(self, candidate: Dict[str, Any], questions: List[str]) -> List[Dict[str, str]]:
        results = []
        for question in questions:
            if not question:
                continue
            results.append({
                "question": question,
                "answer": self.answer_question(candidate, question),
            })
        return results

    def decide_apply(self, candidate: Dict[str, Any], top_jobs: List[Dict[str, Any]], fit_score: float) -> Dict[str, Any]:
        cache_key = ("decide_apply", candidate, top_jobs, fit_score)
        cached = self._get_cached(*cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(candidate, top_jobs, fit_score)

        for agent_response in [self._call_gemini(prompt), self._call_ollama_priority(prompt)]:
            if agent_response:
                parsed = self._extract_json(agent_response)
                if "should_apply" in parsed:
                    return self._set_cached("decide_apply", parsed, candidate, top_jobs, fit_score)

        # Final non-agent fallback: minimal profile-vs-job overlap only when no AI agent is available.
        candidate_skills = set(str(skill).lower() for skill in candidate.get("primary_skills", []) + candidate.get("secondary_skills", []))
        job_overlap = []
        for item in top_jobs:
            job = item.get("job", {})
            job_skills = set(str(skill).lower() for skill in job.get("skills", []))
            overlap = len(candidate_skills & job_skills)
            job_overlap.append((job.get("title", "Unknown"), overlap))

        strongest = max(job_overlap, key=lambda x: x[1], default=("Unknown", 0))
        score = int(max(0, min(100, fit_score)))
        should_apply = score >= 70 or strongest[1] >= 2
        reason = (
            "Strong role and skill alignment with the candidate profile."
            if should_apply
            else "Skill and role fit is weak or below the minimum threshold."
        )
        result = {"should_apply": should_apply, "reason": reason, "fit_score": score}
        return self._set_cached("decide_apply", result, candidate, top_jobs, fit_score)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except Exception:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(stripped[start:end + 1])
                except Exception:
                    pass
        return {
            "should_apply": False,
            "reason": "Local model did not return valid JSON.",
            "fit_score": 0,
        }
