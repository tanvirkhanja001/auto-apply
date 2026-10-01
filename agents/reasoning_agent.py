import json
import os
from typing import Any, Dict, List

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


class ReasoningAgent:
    """Optional local reasoning agent for deciding if a candidate should apply to top jobs."""

    def __init__(self, model_name: str = "qwen2.5:0.5b-instruct", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.remote_available = None
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

    def answer_question(self, candidate: Dict[str, Any], question: str) -> str:
        """Return a short, profile-grounded answer to a screening question."""
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

        # Prefer a quick remote call when the service is reachable, but fall back fast.
        if requests is not None and self.remote_available:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=3,
                )
                if response.ok:
                    answer = response.json().get("response", "").strip()
                    if answer:
                        return answer
            except Exception as ex:
                # Mark remote as unavailable to avoid slow retries later
                print(f"ReasoningAgent: remote model call failed, disabling remote fallback: {ex}")
                self.remote_available = False

        q = (question or "").lower()
        candidate_name = str(candidate.get("name", "Candidate"))
        notice = str(candidate.get("notice_period_days", "30"))
        ctc = str(candidate.get("expected_ctc_lpa", "14"))
        exp = str(candidate.get("total_exp_years", "3"))

        if any(k in q for k in ["notice", "join", "available", "start"]):
            return f"{notice} days"
        if any(k in q for k in ["expected", "ctc", "salary", "package", "lpa"]):
            return f"{ctc} LPA"
        if any(k in q for k in ["experience", "exp", "years"]):
            return f"{exp} years"
        if any(k in q for k in ["java", "spring", "react", "aws", "docker", "microservices", "full stack"]):
            return "Yes, I have hands-on experience in that area."

        # Explicit relocation and live-in-city questions are yes/no prompts, not location answers.
        if any(k in q for k in ["relocate", "ready to relocate", "living in or ready to relocate", "willing to relocate", "open to relocate", "comfortable relocating", "travel", "location"]):
            if any(city in q.lower() for city in ["bengaluru", "bangalore", "pune", "mumbai", "hyderabad", "chennai", "delhi", "remote"]):
                return "Yes"
            if any(k in q for k in ["current location", "current city", "where do you live", "city are you currently in"]):
                return str(candidate.get("current_location", "Nashik"))
            return "Yes"

        if any(k in q for k in ["current location", "current city", "where do you live", "city you live", "city are you currently in"]):
            return str(candidate.get("current_location", "Nashik"))
        if any(k in q for k in ["willing", "open", "comfortable", "eligible"]):
            return "Yes"
        return "Yes, based on my profile and skill fit."

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

    def match_answer_to_options(self, candidate: Dict[str, Any], question: str, answer: str, options: List[str]) -> str:
        """Match the answer to the best option from a list of radio/select options."""
        if not options:
            return answer
        
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
        if requests is not None and self.remote_available:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=3,
                )
                if response.ok:
                    matched = response.json().get("response", "").strip()
                    if matched:
                        # Find the exact option that matches
                        for opt in options:
                            if opt.lower() in matched.lower() or matched.lower() in opt.lower():
                                return opt
                        return matched
            except Exception as ex:
                print(f"ReasoningAgent: remote match failed, using fallback: {ex}")
                self.remote_available = False
        
        # Fallback: simple matching
        answer_lower = answer.lower()
        for opt in options:
            opt_lower = opt.lower()
            if opt_lower == answer_lower or opt_lower in answer_lower or answer_lower in opt_lower:
                return opt
        # Return first option as last resort
        return options[0]

    def decide_apply(self, candidate: Dict[str, Any], top_jobs: List[Dict[str, Any]], fit_score: float) -> Dict[str, Any]:
        prompt = self._build_prompt(candidate, top_jobs, fit_score)

        if requests is not None:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=60,
                )
                if response.ok:
                    text = response.json().get("response", "")
                    return self._extract_json(text)
            except Exception:
                pass

        # Fallback heuristic if the local model is not available.
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
        return {"should_apply": should_apply, "reason": reason, "fit_score": score}

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
