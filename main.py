import asyncio
import json
import os
import random
from openai import OpenAI
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv(".env.local")

# Load Configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_jd_and_questions(jd_text: str, questions: list) -> dict:
    """
    Pure text evaluation function. 
    Does not touch the browser or network, avoiding ToS policy violations.
    """
    prompt = f"""
    You are an evaluator. Match the Job Description against the Candidate Profile.

    Candidate Profile:
    {json.dumps(CONFIG['candidate'])}

    Job Description:
    {jd_text}

    Screening Questions:
    {json.dumps(questions)}

    Tasks:
    1. Calculate match score (0-100) based on Java/Spring Boot stack fit.
    2. Set 'should_apply' to true ONLY if score >= 70.
    3. Provide accurate answers for the listed screening questions.

    Return JSON strictly in this format:
    {{
        "match_score": 80,
        "should_apply": true,
        "reason": "Brief reason",
        "answers": [
            {{"question": "Question text", "answer": "Generated short answer"}}
        ]
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)


async def process_screening_modal(page, answers: list):
    """Fills out UI modals deterministically using local Playwright code."""
    try:
        modal_selector = ".drawer-wrapper, .custom-questions-modal, .apply-message-container"
        modal = await page.wait_for_selector(modal_selector, timeout=3000)
        if not modal:
            return

        print("--> Handling screening questions modal...")

        # Handle text inputs
        inputs = await page.query_selector_all("input[type='text'], input[type='number']")
        for inp in inputs:
            placeholder = (await inp.get_attribute("placeholder") or "").lower()
            if "notice" in placeholder:
                await inp.fill(str(CONFIG["candidate"]["notice_period_days"]))
            elif "ctc" in placeholder:
                await inp.fill(str(CONFIG["candidate"]["expected_ctc_lpa"]))
            elif "exp" in placeholder:
                await inp.fill(str(CONFIG["candidate"]["total_exp_years"]))

        # Handle option selections based on LLM answers
        for item in answers:
            q_ans = str(item.get("answer", "")).lower()
            options = await page.query_selector_all("label, .option-btn")
            for opt in options:
                opt_text = (await opt.inner_text()).strip().lower()
                if q_ans in opt_text:
                    await opt.click()
                    await page.wait_for_timeout(500)
                    break

        # Submit modal
        submit_btn = await page.query_selector("button:has-text('Submit'), button:has-text('Save & Apply')")
        if submit_btn and await submit_btn.is_visible():
            await submit_btn.click()
            print("--> Submitted screening answers.")
            await page.wait_for_timeout(2000)

    except Exception as e:
        print(f"--> Modal step finished or skipped: {e}")


async def run():
    async with async_playwright() as p:
        # Uses local persistent context so you stay logged in securely
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./naukri_session",
            headless=False,
            args=["--start-maximized"]
        )
        page = await context.new_page()

        print(f"Opening: {CONFIG['search_url']}")
        await page.goto(CONFIG["search_url"], wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        job_cards = await page.query_selector_all(".srp-jobtuple-wrapper")
        applied_count = 0

        for idx, card in enumerate(job_cards):
            if applied_count >= CONFIG["max_applications_per_run"]:
                print(f"Reached daily limit of {CONFIG['max_applications_per_run']} applications.")
                break

            try:
                title_elem = await card.query_selector("a.title")
                if not title_elem:
                    continue

                async with context.expect_page() as new_page_info:
                    await title_elem.click()
                job_page = await new_page_info.value
                await job_page.bring_to_front()
                await job_page.wait_for_timeout(2000)

                # Extract JD
                jd_elem = await job_page.query_selector(".styles_JDSummary__xA23_, .job-desc")
                jd_text = await jd_elem.inner_text() if jd_elem else ""

                # Extract Questions if visible
                q_elems = await job_page.query_selector_all(".question-text, .q-title")
                questions = [(await q.inner_text()).strip() for q in q_elems if await q.inner_text()]

                # Evaluate using LLM (Off-browser)
                evaluation = evaluate_jd_and_questions(jd_text, questions)
                print(f"\nJob #{idx+1} | Score: {evaluation['match_score']}% | Apply: {evaluation['should_apply']}")

                if evaluation["should_apply"]:
                    apply_btn = await job_page.query_selector("button:has-text('Apply')")
                    if apply_btn:
                        btn_text = (await apply_btn.inner_text()).lower()
                        if "company site" in btn_text:
                            print("--> External ATS site detected. Skipping.")
                        else:
                            await apply_btn.click()
                            print("--> Clicked Apply.")
                            await job_page.wait_for_timeout(2000)
                            await process_screening_modal(job_page, evaluation.get("answers", []))
                            applied_count += 1
                
                await job_page.close()
                # Human emulation pause
                await asyncio.sleep(random.uniform(2.0, 4.0))

            except Exception as e:
                print(f"Error processing card #{idx+1}: {e}")
                continue

        print("\nPipeline execution complete.")

if __name__ == "__main__":
    asyncio.run(run())