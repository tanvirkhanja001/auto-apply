import warnings
from typing import Dict


class ResumeAgent:
    """Deprecated scaffold.

    The active flow does not require a separate resume-tailoring agent.
    """

    def __init__(self, profile: Dict):
        warnings.warn(
            "ResumeAgent is deprecated; use the active reasoning and matching pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.profile = profile

    def prepare_for_job(self, job: Dict) -> Dict:
        # Minimal transformation: keep original resume but include a short tailoring note
        tailored = dict(self.profile)
        tailored['tailored_for'] = job.get('title')
        return tailored
