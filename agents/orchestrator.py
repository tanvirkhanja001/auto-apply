import asyncio
import warnings
from typing import List, Dict

from agents.matching_agent import MatchingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.questionnaire_agent import QuestionnaireAgent


class Orchestrator:
    """Single active orchestration path for job matching and apply decisions.

    This is the real runtime entry point. The older scaffold agents are kept only for backwards
    compatibility and are deprecated.
    """

    def __init__(self, config: Dict, context=None):
        self.config = config
        self.context = context
        self.candidate = config.get('candidate', {})
        self.matcher = MatchingAgent(model_name="BAAI/bge-small-en-v1.5")
        self.reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
        self.questionnaire = QuestionnaireAgent(self.candidate)

    def rank_jobs(self, jobs: List[Dict], top_n: int = 5):
        return self.matcher.rank_jobs(self.candidate, jobs, top_n=top_n)

    def decide_apply(self, jobs: List[Dict], top_n: int = 5):
        ranked = self.rank_jobs(jobs, top_n=top_n)
        if not ranked:
            return {"should_apply": False, "reason": "No jobs available", "fit_score": 0, "ranked_jobs": []}
        top_jobs = [{"rank": item["rank"], "score": item["score"], "job": item["job"]} for item in ranked]
        fit_score = int(round(sum(item["score"] for item in ranked) / len(ranked) * 100))
        reasoning = self.reasoner.decide_apply(self.candidate, top_jobs, fit_score)
        reasoning["ranked_jobs"] = top_jobs
        reasoning["fit_score"] = reasoning.get("fit_score", fit_score)
        return reasoning

    def answer_questions(self, questions: List[str]):
        if not questions:
            return []
        return [
            {"question": q, "answer": self.questionnaire.reasoner.answer_question(self.candidate, q)}
            for q in questions if q
        ]

    async def run(self, jobs: List[Dict], questions: List[str] | None = None, max_jobs: int = 5):
        ranked = self.rank_jobs(jobs[:max_jobs], top_n=min(max_jobs, len(jobs)))
        if not ranked:
            return {"jobs": [], "decision": {"should_apply": False, "reason": "No jobs available"}}

        decision = self.decide_apply(jobs[:max_jobs], top_n=min(max_jobs, len(jobs)))
        answers = self.answer_questions(questions or [])
        return {
            "jobs": ranked,
            "decision": decision,
            "answers": answers,
        }
