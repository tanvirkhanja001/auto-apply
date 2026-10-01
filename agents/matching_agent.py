import json
from typing import Any, Dict, List

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MatchingAgent:
    """Small local matching engine using BGE-small by default. Falls back to TF-IDF if model is unavailable."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name, device=device)
            except Exception as exc:  # pragma: no cover
                print(f"Warning: unable to load {model_name}: {exc}")
                self.model = None

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return " ".join(str(v) for v in value)
        return str(value)

    def _job_text(self, job: Dict[str, Any]) -> str:
        title = self._normalize_text(job.get("title", ""))
        description = self._normalize_text(job.get("description", ""))
        skills = self._normalize_text(job.get("skills", []))
        return " ".join(part for part in [title, description, skills] if part)

    def _resume_text(self, resume: Dict[str, Any]) -> str:
        headline = self._normalize_text(resume.get("headline", ""))
        summary = self._normalize_text(resume.get("summary", ""))
        skills = self._normalize_text(resume.get("skills", []))
        primary = self._normalize_text(resume.get("primary_skills", []))
        secondary = self._normalize_text(resume.get("secondary_skills", []))
        return " ".join(part for part in [headline, summary, skills, primary, secondary] if part)

    def rank_jobs(self, resume: Dict[str, Any], jobs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not jobs:
            return []

        if self.model is not None:
            try:
                resume_text = self._resume_text(resume)
                job_texts = [self._job_text(job) for job in jobs]
                embeddings = self.model.encode([resume_text] + job_texts, convert_to_numpy=True, normalize_embeddings=True)
                resume_vec = embeddings[0]
                job_vecs = embeddings[1:]
                similarity = cosine_similarity([resume_vec], job_vecs)[0]
                ranked = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)[:top_n]
                result = []
                for index, score in ranked:
                    result.append({
                        "rank": len(result) + 1,
                        "score": round(float(score), 4),
                        "job": jobs[index],
                    })
                return result
            except Exception as exc:  # pragma: no cover
                print(f"Warning: embedding ranking failed; falling back to TF-IDF: {exc}")

        vectorizer = TfidfVectorizer(stop_words="english")
        docs = [self._job_text(job) for job in jobs]
        job_matrix = vectorizer.fit_transform(docs)
        resume_vec = vectorizer.transform([self._resume_text(resume)])
        similarity = cosine_similarity(resume_vec, job_matrix)[0]
        ranked = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)[:top_n]

        result = []
        for index, score in ranked:
            result.append({
                "rank": len(result) + 1,
                "score": round(float(score), 4),
                "job": jobs[index],
            })
        return result

    def to_json(self, resume: Dict[str, Any], jobs: List[Dict[str, Any]], top_n: int = 5) -> str:
        return json.dumps({
            "candidate": resume,
            "top_matches": self.rank_jobs(resume, jobs, top_n=top_n)
        }, indent=2)
