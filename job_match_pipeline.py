import json
from pathlib import Path

from agents.matching_agent import MatchingAgent
from agents.reasoning_agent import ReasoningAgent


def run_pipeline(candidate_path: str, jobs_path: str, top_n: int = 5):
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    jobs = json.loads(Path(jobs_path).read_text(encoding="utf-8"))

    matcher = MatchingAgent(model_name="BAAI/bge-small-en-v1.5")
    ranked = matcher.rank_jobs(candidate, jobs, top_n=top_n)
    if not ranked:
        return {"candidate": candidate, "top_matches": [], "reasoning": {"should_apply": False, "reason": "No jobs loaded", "fit_score": 0}}

    top_jobs = [{"rank": item["rank"], "score": item["score"], "job": item["job"]} for item in ranked]
    fit_score = int(round(sum(item["score"] for item in ranked) / len(ranked) * 100))

    reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
    reasoning = reasoner.decide_apply(candidate, top_jobs, fit_score)

    payload = {
        "candidate": candidate,
        "top_matches": top_jobs,
        "fit_score": fit_score,
        "reasoning": reasoning,
    }
    return payload


if __name__ == "__main__":
    result = run_pipeline("data/tanvir_resume.json", top_n=3)
    print(json.dumps(result, indent=2))
