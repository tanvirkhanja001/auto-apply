from typing import Dict

class ResumeAgent:
    """Select/adapt the resume for the JD. Only use truthful info from resume input."""

    def __init__(self, profile: Dict):
        self.profile = profile

    def prepare_for_job(self, job: Dict) -> Dict:
        # Minimal transformation: keep original resume but include a short tailoring note
        tailored = dict(self.profile)
        tailored['tailored_for'] = job.get('title')
        return tailored
