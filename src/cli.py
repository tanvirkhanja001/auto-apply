import json
import argparse
from pathlib import Path
from .job_matcher import JobMatcher


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True, help="Path to jobs.json")
    p.add_argument("--resume", required=True, help="Path to resume.json")
    p.add_argument("--top", type=int, default=5, help="Top N matches")
    args = p.parse_args()

    jobs = load_json(args.jobs)
    resume = load_json(args.resume)

    matcher = JobMatcher(jobs)
    matches = matcher.top_matches_for_resume(resume, args.top)
    print(json.dumps(matches, indent=2))


if __name__ == "__main__":
    main()
