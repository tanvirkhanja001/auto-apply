import warnings
from typing import Dict, List
from playwright.async_api import Page, BrowserContext


class ApplicationAgent:
    """Deprecated scaffold.

    Browser actions are handled in the active live flow in main.py.
    """

    def __init__(self, context: BrowserContext, candidate: Dict):
        warnings.warn(
            "ApplicationAgent is deprecated; use the live browser flow in main.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.context = context
        self.candidate = candidate

    async def open_and_apply(self, job_url: str, answers: List[Dict]) -> bool:
        page = await self.context.new_page()
        try:
            await page.goto(job_url, wait_until='domcontentloaded')
            await page.wait_for_timeout(2000)
            apply_btn = await page.query_selector("button:has-text('Apply')")
            if apply_btn:
                await apply_btn.click()
                await page.wait_for_timeout(2000)
                # The QuestionnaireAgent should handle filling
                return True
            return False
        finally:
            try:
                await page.close()
            except Exception:
                pass
