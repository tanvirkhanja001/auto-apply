import asyncio
from typing import List, Dict
from agents.job_finder_agent import JobFinderAgent
from agents.job_analyzer_agent import JobAnalyzerAgent
from agents.resume_agent import ResumeAgent
from agents.application_agent import ApplicationAgent
from agents.questionnaire_agent import QuestionnaireAgent

class Orchestrator:
    def __init__(self, config: Dict, context=None):
        self.config = config
        self.context = context
        self.finder = JobFinderAgent(config.get('search', {}))
        self.analyzer = JobAnalyzerAgent()
        self.resume_agent = ResumeAgent(config.get('candidate', {}))
        self.questionnaire = QuestionnaireAgent(config.get('candidate', {}))

    async def run(self, page, max_jobs: int = 5):
        # 1. Find jobs
        jobs = await self.finder.search(page, max_results=max_jobs)

        for job in jobs:
            analysis = self.analyzer.analyze(self.config.get('candidate', {}), job)
            print(f"Job: {job.get('title')} -> {analysis.get('decision')} ({analysis.get('score')}%)")
            if analysis.get('decision') == 'APPLY':
                # 2. Prepare resume
                tailored = self.resume_agent.prepare_for_job(job)
                # 3. Open and apply (ApplicationAgent) - simplified: returns True if apply initiated
                app_agent = ApplicationAgent(self.context, self.config.get('candidate', {}))
                applied = await app_agent.open_and_apply(job.get('url',''), [])
                if applied:
                    print("Apply initiated; running questionnaire detection/fill (local detection only).")
                    # detect questions via page - simplified in main flow where page is available
                    # 4. run questionnaire agent offline if questions available
                    # In live flow, QuestionnaireAgent will be invoked with scraped questions and answers
                else:
                    print("Apply not initiated or external ATS; skipping questionnaire.")

        return True
