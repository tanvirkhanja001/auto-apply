from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobMatcher:
    """Simple job matcher using TF-IDF + cosine similarity."""

    def __init__(self, jobs: List[Dict[str, Any]] = None):
        self.jobs = jobs or []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_vecs = None
        if self.jobs:
            self._fit_jobs(self.jobs)

    def _job_text(self, job: Dict[str, Any]) -> str:
        parts = [job.get("title", ""), job.get("description", "")]
        skills = job.get("skills") or []
        if isinstance(skills, list):
            parts.append(" ".join(skills))
        else:
            parts.append(str(skills))
        return " \n ".join([p for p in parts if p])

    def _resume_text(self, resume: Dict[str, Any]) -> str:
        parts = [resume.get("headline", ""), resume.get("summary", "")]
        skills = resume.get("skills") or []
        if isinstance(skills, list):
            parts.append(" ".join(skills))
        else:
            parts.append(str(skills))
        return " \n ".join([p for p in parts if p])

    def _fit_jobs(self, jobs: List[Dict[str, Any]]):
        self.jobs = jobs
        docs = [self._job_text(j) for j in jobs]
        self.job_vecs = self.vectorizer.fit_transform(docs)

    def load_jobs(self, jobs: List[Dict[str, Any]]):
        """Replace current jobs and re-fit the vectorizer."""
        self._fit_jobs(jobs)

    def top_matches_for_resume(self, resume: Dict[str, Any], top_n: int = 5):
        """Return top N matching jobs for a given resume dict.

        Returns a list of {"score": float, "job": job_dict} ordered by score desc.
        """
        if self.job_vecs is None:
            raise ValueError("No jobs loaded")
        doc = self._resume_text(resume)
        vec = self.vectorizer.transform([doc])
        sims = cosine_similarity(vec, self.job_vecs)[0]
        idxs = sims.argsort()[::-1][:top_n]
        return [{"score": float(sims[i]), "job": self.jobs[i]} for i in idxs]
