import asyncio
import warnings
from typing import List, Dict
from playwright.async_api import Page


class JobFinderAgent:
    """Deprecated scaffold.

    Use MatchingAgent + ReasoningAgent + QuestionnaireAgent in the active flow.
    """

    def __init__(self, search_config: Dict):
        warnings.warn(
            "JobFinderAgent is deprecated; use the active matching/reasoning pipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.search_config = search_config

    async def search(self, page: Page, max_results: int = 50) -> List[Dict]:
        results = []
        q = self.search_config.get("query") or self.search_config.get("role") or ""
        location = self.search_config.get("location") or ""
        url = f"https://www.naukri.com/{''}"

        # Very small, safe navigation: load recommended page and scrape cards
        await page.goto("https://www.naukri.com/mnj/recommendedjobs", wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)
        cards = await page.query_selector_all(".srp-jobtuple-wrapper, .jobTuple, .cust-job-tuple")
        for idx, card in enumerate(cards[:max_results]):
            title = (await (card.query_selector('.title') or card.query_selector('h2') )).inner_text() if (await (card.query_selector('.title') or card.query_selector('h2'))) else f"Job #{idx+1}"
            company_el = await card.query_selector('.company') or await card.query_selector('.name')
            company = (await company_el.inner_text()).strip() if company_el else ''
            link_el = await card.query_selector('a')
            url = await link_el.get_attribute('href') if link_el else ''

            # Try to fetch description via a quick click-open
            job = {
                'title': title,
                'company': company,
                'description': '',
                'location': location,
                'url': url,
            }
            results.append(job)
        return results
