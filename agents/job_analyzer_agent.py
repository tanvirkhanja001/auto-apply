import json
from typing import Dict, List

class JobAnalyzerAgent:
    """Compare JD with resume/profile, identify matched/missing skills, and return relevance and APPLY/SKIP."""

    def __init__(self, model_name: str = "SmolLM2-360M-Instruct"):
        self.model_name = model_name

    def analyze(self, resume: Dict, job: Dict) -> Dict:
        jd = job.get('description', '') or job.get('title','')
        resume_skills = set([s.lower() for s in resume.get('skills',[])])
        jd_skills = set([s.lower() for s in job.get('skills',[])])
        matched = list(resume_skills & jd_skills)
        missing = list(jd_skills - resume_skills)
        # naive relevance score: matched / (matched + missing) * 100
        score = 0
        if matched or missing:
            score = int(len(matched) / max(1, (len(matched) + len(missing))) * 100)
        decision = 'APPLY' if score >= 50 else 'SKIP'
        return {
            'score': score,
            'decision': decision,
            'matched_skills': matched,
            'missing_skills': missing,
            'summary': f'{len(matched)} matched, {len(missing)} missing',
        }
