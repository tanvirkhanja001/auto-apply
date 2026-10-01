import asyncio
import json
import os
import random
import time
from datetime import datetime
from google import genai
from google.genai import types
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv(".env.local")

# Local matching and reasoning agents
from agents.matching_agent import MatchingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.questionnaire_agent import QuestionnaireAgent


SKILL_HINTS = [
    "java", "spring boot", "spring", "hibernate", "jpa", "postgresql", "mysql",
    "mongodb", "react", "next.js", "nextjs", "javascript", "typescript", "aws",
    "docker", "kubernetes", "microservices", "rest api", "rest", "oauth", "jwt",
    "auth0", "sql", "redis", "node.js", "tailwind", "redux", "ci/cd"
]


# 1. Candidate Context Profile & Search Configuration
# Load Configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

def _fallback_decision(match_context: dict | None, jd_text: str, questions: list) -> dict:
    """Local fallback when Gemini is unavailable or disconnects."""
    best_score = 0
    if match_context and match_context.get("local_matcher", {}).get("status") == "ok":
        best_score = int(match_context["local_matcher"].get("average_fit_score", 0))

    local_reasoning = match_context.get("local_reasoning", {}) if match_context else {}
    should_apply = bool(local_reasoning.get("should_apply", best_score >= 70))
    reason = str(local_reasoning.get("reason") or "Fallback local decision based on skill match.")

    reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
    answers = reasoner.build_answers_for_questions(CONFIG["candidate"], questions)
    return {"match_score": best_score, "should_apply": should_apply, "reason": reason, "answers": answers}


def is_application_success_page(text: str | None, url: str | None = None) -> bool:
    """Return True only when the page clearly confirms the application was submitted."""
    haystack_url = (url or "").lower()
    haystack_text = (text or "").lower()
    haystack = " ".join(part for part in [text or "", url or ""] if part).lower()
    
    if not haystack:
        return False

    # Check URL explicitly for success paths
    url_success_markers = [
        "/apply/success",
        "application-status=success",
        "saveapply",
        "/myapply/",
        "myapply",
    ]
    if any(marker in haystack_url for marker in url_success_markers):
        return True

    text_markers = [
        "application submitted",
        "applied successfully",
        "your application has been submitted",
        "submitted successfully",
        "application successful",
        "profile submitted",
        "thank you for applying",
        "thank you for your responses",
        "your profile is submitted",
        "saved successfully",
        "we have received your application",
        "candidate profile saved",
        "successfully applied",
        "application has been sent",
    ]
    
    # "applied to" can be tricky if it appears before the job title on a confirmation page, 
    # but we must ensure we don't trigger it prematurely on a generic "You are about to apply to..." text
    # Let's rely on the stronger markers above instead of "applied to " or "saveapply" URL
    
    return any(marker in haystack_text for marker in text_markers)


def evaluate_jd_and_questions_gemini(jd_text: str, questions: list, match_context: dict | None = None) -> dict:
    """
    Evaluates JD match using Gemini 2.5 Flash and provides answers to screening questions.
    """
    context_block = ""
    if match_context:
        context_block = f"""
        Local job matching context:
        {json.dumps(match_context, indent=2)}
        """

    prompt = f"""
    You are an AI Job Matching Assistant.
    Candidate Profile: {json.dumps(CONFIG['candidate'])}
    Job Description: {jd_text}
    Screening Questions: {json.dumps(questions)}
    {context_block}

    Tasks:
    1. Calculate match score (0-100) based on how well the Job Description matches the Candidate Profile (Java, Spring Boot, React/Next.js, AWS).
    2. Use the local matching context as evidence, but do not override the actual profile-to-job fit logic.
    3. Set 'should_apply' to true ONLY if score >= 70.
    4. Generate short, accurate answers for the screening questions based on the candidate profile.

    Output STRICTLY a valid JSON object with keys:
    "match_score" (int), "should_apply" (bool), "reason" (str), "answers" (list of objects with "question" and "answer").
    """

    if gemini_client is None:
        return _fallback_decision(match_context, jd_text, questions)

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini request failed; using local fallback. Error: {e}")
        return _fallback_decision(match_context, jd_text, questions)


def extract_skill_tokens(text: str) -> list:
    """Very lightweight skill extraction for scraped job description text."""
    if not text:
        return []
    lowered = text.lower()
    found = []
    for skill in SKILL_HINTS:
        if skill in lowered:
            found.append(skill)
    return found


def build_match_context(candidate: dict, jobs: list, top_n: int = 5) -> dict:
    """Generate the compact JSON context for Gemini from the local ranking and reasoning agents."""
    if not jobs:
        return {"local_matcher": {"status": "skipped", "reason": "No job bank loaded"}}

    matcher = MatchingAgent(model_name="BAAI/bge-small-en-v1.5")
    ranked_jobs = matcher.rank_jobs(candidate, jobs, top_n=top_n)
    if not ranked_jobs:
        return {"local_matcher": {"status": "skipped", "reason": "No jobs available for ranking"}}

    top_jobs = [{"rank": item["rank"], "score": item["score"], "job": item["job"]} for item in ranked_jobs]
    fit_score = int(round(sum(item["score"] for item in ranked_jobs) / len(ranked_jobs) * 100))
    reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
    reasoning = reasoner.decide_apply(candidate, top_jobs, fit_score)

    return {
        "local_matcher": {
            "status": "ok",
            "model": "BAAI/bge-small-en-v1.5",
            "top_matches": top_jobs,
            "average_fit_score": fit_score,
        },
        "local_reasoning": reasoning,
    }


async def all_contexts(p):
    """Yield the page and any frame contexts in a safe, iterable way."""
    seen = set()
    yield p
    seen.add(id(p))
    try:
        frames = list(getattr(p, "frames", []) or [])
    except Exception:
        frames = []
    for frame in frames:
        if id(frame) in seen:
            continue
        seen.add(id(frame))
        yield frame


async def _read_chatbot_question(drawer) -> str:
    """Extract the latest question text from the Naukri chatbot drawer.

    Naukri shows questions in bot-message bubbles.  We look for the last
    visible bot message that contains a question-like string.
    """
    bot_msg_selectors = [
        ".botMsg",                       # primary chatbot bubble class
        ".msg-bot",                      # alternate
        ".chatbot_message--bot",         # newer variant
        ".chatbot-bot-msg",
        ".bot-message",
        "[class*='botMsg']",
        "[class*='bot-msg']",
    ]

    question = ""
    for sel in bot_msg_selectors:
        try:
            elems = await drawer.query_selector_all(sel)
            if not elems:
                continue
            print(f"    --> _read_chatbot_question: selector '{sel}' found {len(elems)} elements")
            # Take the last bot message (the current question)
            for el in reversed(elems):
                try:
                    if not await el.is_visible():
                        continue
                    txt = (await el.inner_text() or "").strip()
                    if txt and len(txt) > 5:
                        question = " ".join(txt.split())
                        print(f"    --> _read_chatbot_question: found question via '{sel}': '{question[:100]}'")
                        break
                except Exception:
                    continue
            if question:
                break
        except Exception:
            continue

    # Fallback: look for any visible paragraph/span with a question mark inside the drawer
    if not question:
        # Also check for common Naukri question containers (problem-of-the-day, mcq cards)
        container_selectors = [
            ".problem-of-the-day",
            ".question-card",
            ".mcq-question",
            ".questionTitle",
            ".q-card",
        ]
        for csel in container_selectors:
            try:
                elems = await drawer.query_selector_all(csel)
                for el in reversed(elems):
                    try:
                        if not await el.is_visible():
                            continue
                        txt = (await el.inner_text() or "").strip()
                        if txt and ("?" in txt or len(txt) > 10):
                            question = " ".join(txt.split())
                            break
                    except Exception:
                        continue
                if question:
                    break
            except Exception:
                continue

        if not question:
            for sel in ["p", "span", "div"]:
                try:
                    elems = await drawer.query_selector_all(sel)
                    for el in reversed(elems):
                        try:
                            if not await el.is_visible():
                                continue
                            txt = (await el.inner_text() or "").strip()
                            if txt and "?" in txt and len(txt) > 10:
                                question = " ".join(txt.split())
                                break
                        except Exception:
                            continue
                    if question:
                        break
                except Exception:
                    continue

    return question


# Helper function for local radio option matching
def _match_answer_to_radio_options(answer: str, normalized: str, options: list) -> str | None:
    """Local fallback to match answer to radio options."""
    if not options:
        return None
    
    import re
    answer_lower = answer.lower()
    normalized_lower = normalized.lower()
    
    # Numeric range matching for experience
    answer_nums = re.findall(r'\d+', answer)
    answer_num = int(answer_nums[0]) if answer_nums else None
    
    for opt in options:
        opt_lower = opt.lower()
        
        # Exact or substring match
        if opt_lower == answer_lower or opt_lower in answer_lower or answer_lower in opt_lower:
            return opt
        
        if opt_lower == normalized_lower or opt_lower in normalized_lower or normalized_lower in opt_lower:
            return opt
        
        # Numeric range matching
        if answer_num is not None:
            range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', opt_lower)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
                if low <= answer_num <= high:
                    return opt
            
            plus_match = re.search(r'(\d+)\s*\+', opt_lower)
            if plus_match:
                low = int(plus_match.group(1))
                if answer_num >= low:
                    return opt
            
            # "X days" matching to "X months" (30 days ≈ 1 month)
            if "day" in answer_lower and "month" in opt_lower:
                month_match = re.search(r'(\d+)', opt_lower)
                if month_match:
                    months = int(month_match.group(1))
                    if months == 1 and answer_num <= 45:  # 30 days ≈ 1 month
                        return opt
                    if months > 1 and answer_num >= months * 30:
                        return opt
            
            # "X months" matching to "X days"
            if "month" in answer_lower and "day" in opt_lower:
                month_match = re.search(r'(\d+)', answer_lower)
                day_match = re.search(r'(\d+)', opt_lower)
                if month_match and day_match:
                    months = int(month_match.group(1))
                    days = int(day_match.group(1))
                    if months * 30 >= days:
                        return opt
    
    return None


async def _answer_in_chatbot_drawer(drawer, answer: str, question: str = "", reasoner=None) -> bool:
    """Fill the current input control inside the Naukri chatbot drawer and return True if filled.

    Handles three Naukri chatbot question types:
      1. Radio buttons (Yes/No or multiple choice)
      2. Text/number input field ("Type message here...")
      3. Select dropdown
    """
    normalized = QuestionnaireAgent.normalize_field_answer("radio", answer)
    filled = False

    # NEW: Prefer explicit input/select/textarea controls inside the drawer and handle by control type.
    try:
        inputs = await drawer.query_selector_all("input")
        print(f"    --> Found {len(inputs)} input elements in drawer")
    except Exception:
        inputs = []

    # Process input elements first (by type)
    if inputs:
        # Group radios by name so we can choose the best match collectively
        radios_by_name = {}
        for inp in inputs:
            try:
                if not await inp.is_visible():
                    continue
                itype = (await inp.get_attribute("type") or "").lower()
                name = (await inp.get_attribute("name") or "").strip()
                print(f"    --> Input found: type='{itype}', name='{name}'")

                if itype == "radio":
                    radios_by_name.setdefault(name or "__noname__", []).append(inp)
                    continue

                if itype == "checkbox":
                    try:
                        desired = normalized in {"yes", "true", "1"} or answer.strip().lower() in {"yes", "true", "1"}
                        try:
                            is_checked = await inp.is_checked()
                        except Exception:
                            is_checked = False
                        if desired != is_checked:
                            await inp.click()
                        print(f"    --> Toggled checkbox (name={name}) to {desired} for answer '{answer}'")
                        return True
                    except Exception:
                        continue

                # Text-like inputs
                if itype in {"text", "search", "email", "number", "tel", "password", "date"} or itype == "":
                    try:
                        await inp.click()
                        await inp.fill("")
                        await inp.fill(answer)
                        print(f"    --> Filled input[type={itype}] name={name} with '{answer}'")
                        return True
                    except Exception:
                        continue
            except Exception:
                continue

        # Handle radio groups: for each group, pick the radio whose label matches the normalized answer
        print(f"    --> radios_by_name groups: {list(radios_by_name.keys())}")
        for name, radios in radios_by_name.items():
            try:
                print(f"    --> Processing radio group '{name}' with {len(radios)} options")
                print(f"    --> Answer: '{answer}', Normalized: '{normalized}'")
                
                # Collect all radio labels/values for matching
                radio_options = []
                for r in radios:
                    try:
                        label_text = ""
                        try:
                            parent = await r.evaluate_handle("el => el.closest('label')")
                            if parent:
                                label_text = (await parent.inner_text() or "").strip()
                        except Exception:
                            pass
                        
                        if not label_text:
                            try:
                                label_text = await r.evaluate("""el => {
                                    let label = el.nextElementSibling;
                                    while (label && label.tagName !== 'LABEL') label = label.nextElementSibling;
                                    return label ? label.innerText : '';
                                }""") or ""
                                label_text = label_text.strip()
                            except Exception:
                                pass
                        
                        if not label_text:
                            try:
                                label_text = await r.evaluate("el => el.parentElement ? el.parentElement.innerText : ''") or ""
                                label_text = label_text.strip()
                            except Exception:
                                pass
                        
                        value = (await r.get_attribute("value") or "").strip()
                        display_text = label_text if label_text else value
                        if display_text:
                            radio_options.append(display_text)
                            print(f"      --> Radio option: '{display_text}'")
                    except Exception:
                        continue
                
                # Use ReasoningAgent to match answer to options if available
                matched_option = None
                if reasoner and question and radio_options:
                    print(f"    --> Using ReasoningAgent to match '{answer}' to options: {radio_options}")
                    matched_option = reasoner.match_answer_to_options(CONFIG["candidate"], question, answer, radio_options)
                    print(f"    --> ReasoningAgent matched: '{matched_option}'")
                
                # Fallback to local matching if no reasoner match
                if not matched_option:
                    matched_option = _match_answer_to_radio_options(answer, normalized, radio_options)
                    print(f"    --> Local fallback matched: '{matched_option}'")
                
                # Click the matched radio
                if matched_option:
                    for r in radios:
                        try:
                            # Check if this radio matches our selected option
                            label_text = ""
                            try:
                                parent = await r.evaluate_handle("el => el.closest('label')")
                                if parent:
                                    label_text = (await parent.inner_text() or "").strip().lower()
                            except Exception:
                                pass
                            value = (await r.get_attribute("value") or "").strip().lower()
                            
                            if matched_option.lower() in label_text or matched_option.lower() in value or label_text in matched_option.lower() or value in matched_option.lower():
                                print(f"      --> Attempting click on radio with label='{label_text}' value='{value}' for matched option '{matched_option}'")
                                try:
                                    await r.click()
                                    print(f"    --> Selected radio in group '{name}' with label '{label_text}' value='{value}' for answer '{answer}'")
                                    return True
                                except Exception as click_e:
                                    print(f"      --> Click failed: {click_e}, trying JS click")
                                    try:
                                        await r.evaluate("el => el.click()")
                                        print(f"    --> Selected radio via JS in group '{name}' with label '{label_text}' value='{value}' for answer '{answer}'")
                                        return True
                                    except Exception as js_e:
                                        print(f"      --> JS click also failed: {js_e}")
                                        continue
                        except Exception:
                            continue
            except Exception:
                continue

    # Next prefer selects and textareas inside drawer
    try:
        selects = await drawer.query_selector_all("select")
        print(f"    --> Found {len(selects)} select elements in drawer")
    except Exception:
        selects = []
    for sel in selects:
        try:
            if not await sel.is_visible():
                continue
            try:
                await sel.select_option(label=answer)
                print(f"    --> Selected <select> option by label '{answer}'")
                return True
            except Exception:
                try:
                    await sel.select_option(value=answer)
                    print(f"    --> Selected <select> option by value '{answer}'")
                    return True
                except Exception:
                    continue
        except Exception:
            continue

    try:
        textareas = await drawer.query_selector_all("textarea")
        print(f"    --> Found {len(textareas)} textarea elements in drawer")
    except Exception:
        textareas = []
    for ta in textareas:
        try:
            if not await ta.is_visible():
                continue
            await ta.click()
            await ta.fill("")
            await ta.fill(answer)
            print(f"    --> Typed into textarea: '{answer}'")
            return True
        except Exception:
            continue

    # Handle contenteditable fields and role=textbox (chat inputs often use these)
    try:
        editables = await drawer.query_selector_all("[contenteditable='true'], [role='textbox']")
        print(f"    --> Found {len(editables)} contenteditable/textbox elements in drawer")
    except Exception:
        editables = []
    for ed in editables:
        try:
            if not await ed.is_visible():
                continue
            try:
                await ed.click()
            except Exception:
                pass
            try:
                # Clear existing content then type the answer
                await ed.evaluate("el => { if (el.isContentEditable) el.innerText = ''; else if ('value' in el) el.value = ''; }")
            except Exception:
                pass
            try:
                await ed.type(answer)
                print(f"    --> Typed into contenteditable/textbox: '{answer}'")
                return True
            except Exception:
                # last resort: set innerText via JS
                try:
                    await ed.evaluate(f"el => {{ el.innerText = {json.dumps(answer)}; }}")
                    print(f"    --> Set contenteditable innerText for '{answer}'")
                    return True
                except Exception:
                    continue
        except Exception:
            continue

    # If we reach here no explicit controls succeeded — fall back to older heuristics below.
    print(f"    --> Explicit controls failed, trying fallback heuristics...")

    # --- 1. Radio buttons ---------------------------------------------------
    radio_selectors = [
        "input[type='radio']",
        "[role='radio']",
        ".radio-btn input",
        ".chatbot_inputbox input[type='radio']",
    ]
    for sel in radio_selectors:
        try:
            radios = await drawer.query_selector_all(sel)
            if not radios:
                continue
            for radio in radios:
                try:
                    if not await radio.is_visible():
                        continue
                    # Get the label for this radio
                    label_text = ""
                    try:
                        # Check the parent label or adjacent label/span
                        parent = await radio.evaluate_handle("el => el.closest('label') || el.parentElement")
                        label_text = ((await parent.inner_text()) or "").strip().lower()
                    except Exception:
                        try:
                            value = (await radio.get_attribute("value") or "").strip().lower()
                            label_text = value
                        except Exception:
                            pass

                    if not label_text:
                        continue

                    # Match answer to radio option
                    if (normalized == "yes" and "yes" in label_text) or \
                       (normalized == "no" and "no" in label_text) or \
                       (normalized in label_text) or \
                       (label_text in normalized):
                        await radio.click()
                        print(f"    --> Selected radio: '{label_text}' for answer '{answer}'")
                        filled = True
                        break
                except Exception:
                    continue
            if filled:
                break
        except Exception:
            continue

    if filled:
        return True

    # Additional heuristic: if not filled and answer is a yes/no, try clicking visible labels/buttons with that text
    try:
        yn = normalized
        if yn in {"yes", "no"}:
            # search for clickable elements inside drawer containing yes/no
            # Try option-like clickable elements that often render choices
            option_selectors = [
                ".option", ".choice", ".mcq-option", ".answer-choice", ".q-option",
                "label", "button", "span", "div"
            ]
            candidates = []
            for s in option_selectors:
                try:
                    els = await drawer.query_selector_all(s)
                    candidates.extend(els)
                except Exception:
                    continue

            for c in candidates:
                try:
                    if not await c.is_visible():
                        continue
                    txt = ((await c.inner_text()) or "").strip().lower()
                    if not txt:
                        continue
                    # match exact words or contained words
                    if yn == txt or yn in txt or txt in yn:
                        try:
                            await c.click()
                            print(f"    --> Heuristic clicked option element with text '{txt}' for yes/no answer '{answer}'")
                            return True
                        except Exception:
                            continue
                except Exception:
                    continue
    except Exception:
        pass

    # --- 2. Text / number input ----------------------------------------------
    text_selectors = [
        ".chatbot_inputbox input[type='text']",
        ".chatbot_inputbox input[type='number']",
        ".chatbot_inputbox textarea",
        "input[placeholder*='message']",
        "input[placeholder*='type']",
        "input[placeholder*='answer']",
        "textarea[placeholder*='message']",
        "textarea[placeholder*='type']",
        "[contenteditable='true']",
        "[role='textbox']",
        "input[type='text']",
        "input[type='number']",
        "textarea",
    ]
    for sel in text_selectors:
        try:
            field = await drawer.query_selector(sel)
            if field and await field.is_visible():
                await field.click()
                await field.fill("")  # clear first
                await field.fill(answer)
                print(f"    --> Typed answer: '{answer}'")
                return True
        except Exception:
            continue

    # --- 3. Select dropdown --------------------------------------------------
    try:
        select = await drawer.query_selector("select")
        if select and await select.is_visible():
            try:
                await select.select_option(label=answer)
                print(f"    --> Selected dropdown: '{answer}'")
                return True
            except Exception:
                try:
                    await select.select_option(value=answer)
                    return True
                except Exception:
                    pass
    except Exception:
        pass

    return False


async def _click_save_in_drawer(drawer, page) -> bool:
    """Click the Save/Submit button inside the chatbot drawer. Returns True on success."""
    # Naukri-specific save button selectors (more specific first)
    save_selectors = [
        # Naukri specific selectors observed in UI
        "button.chatbot_submit",
        "button[class*='chatbot']",
        "button[class*='save-btn']",
        "button[class*='submit-btn']",
        "button[class*='primary-btn']",
        # Naukri chatbot specific - send message button
        "button[id*='sendMsgbtn']",
        "button[id*='sendMsgbtn_container']",
        "[id*='sendMsgbtn']",
        "[id*='sendMsgbtn_container']",
        # Generic but common patterns
        "button:has-text('Save')",
        "button:has-text('Submit')",
        "button:has-text('Next')",
        "button:has-text('Continue')",
        "button[class*='submit']",
        "button[class*='save']",
        "input[type='submit']",
        # Role-based
        "[role='button']:has-text('Save')",
        "[role='button']:has-text('Submit')",
        "[role='button']:has-text('Next')",
    ]
    
    print(f"    --> Searching for Save button with {len(save_selectors)} selectors...")
    
    # Try all selectors in parallel-ish manner with shorter wait
    wait_ms = 2000  # Reduced from 5000ms
    poll = 200
    
    async def try_click_button(btn, context_name: str) -> bool:
        """Try to click a button with minimal wait for enable."""
        try:
            # Check if button is already in "saved/submitted" state
            try:
                txt = (await btn.inner_text() or "").strip().lower()
                cls = (await btn.get_attribute("class") or "").strip().lower()
                if any(kw in txt for kw in ["saved", "submitted", "applied", "done", "completed"]) or \
                   any(kw in cls for kw in ["saved", "submitted", "applied", "success", "done"]):
                    print(f"    --> {context_name} button already in saved state: text='{txt}', class='{cls}'")
                    return True
            except Exception:
                pass
            
            # Quick enable check - don't wait long
            enabled = False
            try:
                if await btn.is_enabled():
                    enabled = True
                else:
                    # Brief wait only if disabled
                    waited = 0
                    while waited < wait_ms:
                        await page.wait_for_timeout(poll)
                        waited += poll
                        try:
                            if await btn.is_enabled():
                                enabled = True
                                break
                        except Exception:
                            pass
            except Exception:
                try:
                    disabled_attr = await btn.get_attribute("disabled")
                    if not disabled_attr:
                        enabled = True
                except Exception:
                    enabled = True
            
            if not enabled:
                print(f"    --> {context_name} Save/Submit found but remained disabled after {wait_ms}ms.")
                return False
            
            try:
                await btn.click()
                print(f"    --> Clicked Save/Submit in {context_name}.")
                return True
            except Exception as e:
                try:
                    await btn.evaluate("el => el.click()")
                    print(f"    --> Clicked Save/Submit in {context_name} via JS.")
                    return True
                except Exception as e2:
                    print(f"    --> Failed clicking Save/Submit in {context_name}: {e} | {e2}")
                    return False
        except Exception:
            return False
    
    # First, try drawer-level buttons
    print(f"    --> Searching for Save button in drawer with {len(save_selectors)} selectors...")
    for sel in save_selectors:
        try:
            btn = await drawer.query_selector(sel)
            if btn and await btn.is_visible():
                txt = (await btn.inner_text() or "").strip()
                cls = (await btn.get_attribute("class") or "").strip()
                print(f"    --> Found visible button with selector '{sel}': text='{txt}', class='{cls}'")
                if await try_click_button(btn, "drawer"):
                    return True
            elif btn:
                print(f"    --> Selector '{sel}' matched but button not visible")
        except Exception as e:
            print(f"    --> Selector '{sel}' error: {e}")
            continue
    
    # Fallback: try page-level buttons (but only visible ones)
    print(f"    --> Fallback: searching for Save button on page...")
    for sel in save_selectors:
        try:
            btn = await page.query_selector(sel)
            if btn and await btn.is_visible():
                txt = (await btn.inner_text() or "").strip()
                cls = (await btn.get_attribute("class") or "").strip()
                print(f"    --> Found visible button on page with selector '{sel}': text='{txt}', class='{cls}'")
                if await try_click_button(btn, "page"):
                    return True
            elif btn:
                print(f"    --> Page selector '{sel}' matched but button not visible")
        except Exception as e:
            print(f"    --> Page selector '{sel}' error: {e}")
            continue
    
    # Last resort: try clicking any visible button in drawer that looks like submit
    print(f"    --> Last resort: heuristic search for submit-like buttons in drawer...")
    try:
        all_buttons = await drawer.query_selector_all("button, [role='button'], input[type='submit'], input[type='button']")
        print(f"    --> Found {len(all_buttons)} total buttons in drawer, checking for submit-like...")
        for btn in all_buttons:
            try:
                if not await btn.is_visible():
                    continue
                txt = (await btn.inner_text() or "").strip().lower()
                val = (await btn.get_attribute("value") or "").strip().lower()
                aria = (await btn.get_attribute("aria-label") or "").strip().lower()
                cls = (await btn.get_attribute("class") or "").strip().lower()
                # Check if button looks like a submit/save/next button
                if any(kw in txt for kw in ["save", "submit", "next", "continue", "done", "proceed"]) or \
                   any(kw in val for kw in ["save", "submit", "next", "continue"]) or \
                   any(kw in aria for kw in ["save", "submit", "next", "continue"]) or \
                   any(kw in cls for kw in ["submit", "save", "primary", "confirm", "next"]):
                    print(f"    --> Heuristic match: text='{txt}', value='{val}', aria='{aria}', class='{cls}'")
                    if await try_click_button(btn, "drawer (heuristic)"):
                        return True
            except Exception:
                continue
    except Exception as e:
        print(f"    --> Heuristic search error: {e}")
        pass
    
    print(f"    --> No Save/Submit button found in drawer or page")
    return False


async def _handle_standard_form(page, answers: list) -> bool:
    """Fallback handler for standard HTML form-style questionnaires."""
    print("--> No chatbot drawer found. Looking for standard form questionnaire...")
    try:
        q_selectors = [".question-text", ".q-title", ".custom-question", ".question", "label"]
        questions = []
        for sel in q_selectors:
            for el in await page.query_selector_all(sel):
                if await el.is_visible():
                    txt = (await el.inner_text() or "").strip()
                    if txt and "?" in txt and len(txt) > 10:
                        questions.append((el, txt))

        if not questions:
            return False

        print(f"--> Found {len(questions)} standard form questions.")
        reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
        
        for el, question in questions:
            answer = ""
            try:
                answer = reasoner.answer_question(CONFIG["candidate"], question)
            except Exception:
                answer = "Yes"

            print(f"    [Q] {question}\n    [A] {answer}")
            
            try:
                parent = await el.evaluate_handle("el => el.closest('div') || el.parentElement")
                inputs = await parent.query_selector_all("input, textarea, select")
                if not inputs:
                    continue
                
                target = inputs[0]
                tag = (await target.evaluate("el => el.tagName")).lower()
                
                if tag == "select":
                    try:
                        await target.select_option(label=answer)
                    except:
                        try:
                            await target.select_option(value=answer)
                        except:
                            pass
                elif tag == "input" and (await target.get_attribute("type") or "").lower() in ["radio", "checkbox"]:
                    norm_ans = QuestionnaireAgent.normalize_field_answer("radio", answer)
                    for radio in inputs:
                        val = ((await radio.get_attribute("value")) or "").lower()
                        if norm_ans in val or val in norm_ans or (norm_ans == "yes" and "yes" in val) or (norm_ans == "no" and "no" in val):
                            await radio.click()
                            break
                else:
                    await target.fill(answer)
            except Exception as e:
                print(f"    --> Could not fill: {e}")

        # Click save/submit for the form
        for sel in ["button:has-text('Save')", "button:has-text('Submit')", "button:has-text('Apply')", "input[type='submit']"]:
            try:
                btn = await page.query_selector(sel)
                if btn and await btn.is_visible():
                    await btn.click()
                    print("    --> Clicked Submit on form.")
                    await page.wait_for_timeout(1500)  # Reduced from 3000ms
                    break
            except Exception:
                pass

        try:
            if is_application_success_page(await page.content(), getattr(page, "url", "")):
                return True
        except Exception:
            pass

    except Exception as e:
        print(f"--> Error handling form questionnaire: {e}")
        
    return False


async def process_screening_modal(page, answers: list) -> bool:
    """Handle Naukri's chatbot-style screening questionnaire drawer.

    Naukri presents screening questions one at a time in a conversational
    chatbot drawer on the right side of the page.  The flow is:
      1. Read the current question from the bot message bubble.
      2. Determine the answer (from pre-computed answers list or ReasoningAgent).
      3. Fill the radio / text input.
      4. Click 'Save' to advance to the next question.
      5. Repeat until no new question appears or a success marker is detected.
    """
    # ---------- 1. Locate the chatbot drawer ----------
    drawer_selectors = [
        ".chatbot_drawer",
        ".chatbot_container",
        ".screeningQuestions",
        ".screening-questions",
        ".drawer-wrapper",
        ".chatbot-wrapper",
        "[class*='chatbot']",
        "[class*='screening']",
        ".apply-message-container",
        "[role='dialog']",
        # Additional Naukri-specific selectors
        ".naukri-chatbot",
        ".chat-bot",
        ".bot-drawer",
        "[data-testid*='chatbot']",
        "[data-testid*='screening']",
    ]

    drawer = None
    print(f"    --> Searching for drawer with {len(drawer_selectors)} selectors...")
    for sel in drawer_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                drawer = el
                cls = await el.get_attribute("class") or ""
                print(f"    --> Found visible drawer with selector '{sel}': class='{cls}'")
                break
            elif el:
                cls = await el.get_attribute("class") or ""
                print(f"    --> Selector '{sel}' matched but not visible: class='{cls}'")
        except Exception as e:
            print(f"    --> Selector '{sel}' error: {e}")
            continue

    if not drawer:
        # Try inside frames
        print("    --> Drawer not found on main page, checking frames...")
        try:
            for frame in page.frames:
                for sel in drawer_selectors:
                    try:
                        el = await frame.query_selector(sel)
                        if el:
                            drawer = el
                            break
                    except Exception:
                        continue
                if drawer:
                    break
        except Exception:
            pass

    if not drawer:
        print("    --> No drawer found, trying standard form handler...")
        return await _handle_standard_form(page, answers)

    print("--> Found screening questionnaire drawer. Starting Q&A loop...")

    # Build a lookup dict from pre-computed answers
    answer_lookup = {}
    for item in (answers or []):
        q = str(item.get("question", "") or "").strip().lower()
        a = str(item.get("answer", "") or "").strip()
        if q and a:
            answer_lookup[q] = a

    # Create a reasoning agent for answering unseen questions on the fly
    reasoner = ReasoningAgent(model_name="qwen2.5:0.5b-instruct")
    max_questions = 15  # safety cap to avoid infinite loops
    answered_count = 0
    last_question = ""
    stale_rounds = 0

    for _round in range(max_questions):
        # Wait for the next question to render - reduced from 1500ms
        await page.wait_for_timeout(800)

        # Check for success/completion
        try:
            page_text = await page.content()
            current_url = getattr(page, "url", "")
            if is_application_success_page(page_text, current_url):
                print("--> Application submitted successfully!")
                return True
        except Exception:
            pass

        # Check if drawer is still visible
        try:
            still_visible = await drawer.is_visible()
            if not still_visible:
                print("--> Drawer closed. Checking if application was submitted...")
                try:
                    page_text = await page.content()
                    current_url = getattr(page, "url", "")
                    if is_application_success_page(page_text, current_url):
                        return True
                except Exception:
                    pass
                # Drawer closed but no success marker – might still be okay
                return answered_count > 0
        except Exception:
            # Drawer element became stale – likely page navigated
            return answered_count > 0

        # Read the current question
        question = await _read_chatbot_question(drawer)
        if not question:
            # No question found – try reading from the page-level drawer
            question = await _read_chatbot_question(page)

        if not question:
            print(f"    (round {_round + 1}) No question text found in drawer.")
            stale_rounds += 1
            if stale_rounds >= 5:
                print("--> No new questions after 5 rounds. Stopping.")
                break
            continue

        # Detect stale (same question repeated means Save didn't work)
        if question == last_question:
            stale_rounds += 1
            if stale_rounds >= 5:
                print(f"--> Same question repeated {stale_rounds} times. Stopping.")
                break
        else:
            stale_rounds = 0
            last_question = question

        print(f"\n    [Q{answered_count + 1}] {question}")

        # ---------- Find the best answer ----------
        answer = ""
        q_lower = question.lower()

        # Try exact match from pre-computed answers
        for lookup_q, lookup_a in answer_lookup.items():
            if lookup_q in q_lower or q_lower in lookup_q:
                answer = lookup_a
                break

        # Fuzzy keyword match from pre-computed answers
        if not answer:
            q_words = set(q_lower.split())
            best_overlap = 0
            for lookup_q, lookup_a in answer_lookup.items():
                overlap = len(q_words & set(lookup_q.split()))
                if overlap > best_overlap and overlap >= 2:
                    best_overlap = overlap
                    answer = lookup_a

        # Fall back to the ReasoningAgent
        if not answer:
            try:
                answer = reasoner.answer_question(CONFIG["candidate"], question)
            except Exception:
                answer = "Yes"

        print(f"    [A] {answer}")

        # ---------- Fill the answer ----------
        filled = await _answer_in_chatbot_drawer(drawer, answer, question, reasoner)
        if not filled:
            # Try on the page level as a fallback
            filled = await _answer_in_chatbot_drawer(page, answer, question, reasoner)

        # If still not filled, attempt 2 quick retries (some drawers need a small delay)
        if not filled:
            for retry in range(2):
                await page.wait_for_timeout(700)
                filled = await _answer_in_chatbot_drawer(drawer, answer, question, reasoner)
                if filled:
                    break
                filled = await _answer_in_chatbot_drawer(page, answer, question, reasoner)
                if filled:
                    break

        if not filled:
            print(f"    --> Could not fill answer for: {question} after retries.")
            # Save artifact and abort this apply to avoid submitting empty answers
            try:
                os.makedirs("artifacts", exist_ok=True)
                ts = int(time.time())
                qsafe = "".join([c for c in question if c.isalnum() or c.isspace()])[:80].strip().replace(" ", "_")
                fname_png = f"artifacts/failed_fill_{ts}_{qsafe}.png"
                try:
                    await page.screenshot(path=fname_png, full_page=True)
                    fname = fname_png
                except Exception:
                    try:
                        html = await page.content()
                        fname_html = f"artifacts/failed_fill_{ts}_{qsafe}.html"
                        with open(fname_html, "w", encoding="utf-8") as hf:
                            hf.write(html)
                        fname = fname_html
                    except Exception:
                        fname = None
                with open("artifacts/failed_fills.log", "a", encoding="utf-8") as lf:
                    lf.write(f"{datetime.now().isoformat()} question={question} url={getattr(page,'url','')} screenshot={fname}\n")
                print(f"    --> Saved artifact: {fname}")
            except Exception as e:
                print(f"    --> Failed saving artifact for failed fill: {e}")
            return False

        # ---------- Click Save (only when filled) ----------
        saved = await _click_save_in_drawer(drawer, page)
        if not saved:
            print("    --> Could not find Save button.")
            stale_rounds += 1
            if stale_rounds >= 2:
                # Save artifact and abort to avoid infinite loop
                try:
                    os.makedirs("artifacts", exist_ok=True)
                    ts = int(time.time())
                    qsafe = "".join([c for c in question if c.isalnum() or c.isspace()])[:80].strip().replace(" ", "_")
                    fname_png = f"artifacts/failed_fill_after_save_{ts}_{qsafe}.png"
                    try:
                        await page.screenshot(path=fname_png, full_page=True)
                        fname = fname_png
                    except Exception:
                        try:
                            html = await page.content()
                            fname_html = f"artifacts/failed_fill_after_save_{ts}_{qsafe}.html"
                            with open(fname_html, "w", encoding="utf-8") as hf:
                                hf.write(html)
                            fname = fname_html
                        except Exception:
                            fname = None
                    with open("artifacts/failed_fills.log", "a", encoding="utf-8") as lf:
                        lf.write(f"{datetime.now().isoformat()} save_not_found question={question} url={getattr(page,'url','')} screenshot={fname}\n")
                    print(f"    --> Saved artifact for missing Save button: {fname}")
                except Exception as e:
                    print(f"    --> Failed saving artifact for missing Save button: {e}")
                return False
        else:
            # Verify that the Save advanced the drawer (new question) or led to success
            await page.wait_for_timeout(600)  # Reduced from 1200ms
            try:
                still_vis = await drawer.is_visible()
            except Exception:
                still_vis = False

            if not still_vis:
                # Drawer closed — IMMEDIATELY check URL for Naukri success patterns
                try:
                    current_url = getattr(page, "url", "") or ""
                    low_url = current_url.lower()
                    
                    # Naukri redirects to saveApply/myapply on successful submit - treat as instant success
                    if "saveapply" in low_url or "/myapply/" in low_url or "myapply" in low_url:
                        print(f"    --> Detected apply redirect URL: {current_url}; treating as submitted.")
                        return True
                    
                    # Also check page content for immediate success markers
                    try:
                        page_text = await page.content()
                        if is_application_success_page(page_text, current_url):
                            print("    --> Drawer closed after Save and application success detected.")
                            return True
                        low_text = (page_text or "").lower()
                        if "applied to" in low_text or "applied successfully" in low_text or "saved successfully" in low_text or "application submitted" in low_text:
                            print("    --> Found success text after drawer close.")
                            return True
                    except Exception:
                        pass
                    
                    # If not immediately detected, wait briefly for any delayed navigation
                    print("    --> Drawer closed, waiting briefly for potential redirect...")
                    wait_attempts = 5  # Reduced from 10 (5 * 500ms = 2.5s max)
                    for _poll in range(wait_attempts):
                        await page.wait_for_timeout(500)
                        try:
                            current_url = getattr(page, "url", "") or ""
                            low_url = current_url.lower()
                            if "saveapply" in low_url or "/myapply/" in low_url or "myapply" in low_url:
                                print(f"    --> Detected apply redirect URL after wait: {current_url}; treating as submitted.")
                                return True
                            page_text = await page.content()
                            if is_application_success_page(page_text, current_url):
                                print("    --> Application success detected after wait.")
                                return True
                            low_text = (page_text or "").lower()
                            if "applied to" in low_text or "applied successfully" in low_text or "saved successfully" in low_text or "application submitted" in low_text:
                                print("    --> Found success text after wait.")
                                return True
                        except Exception:
                            pass

                    # If we reach here no success detected — but drawer closed after save, 
                    # this might still be a successful submit (some flows don't show explicit confirmation)
                    # Log and return True if we answered at least one question
                    print("    --> Drawer closed after Save but no explicit success marker. Assuming submit may have succeeded.")
                    return answered_count > 0
                except Exception:
                    return answered_count > 0
            else:
                    # Drawer still visible — check that the question advanced
                    try:
                        new_q = await _read_chatbot_question(drawer)
                    except Exception:
                        new_q = ""
                    if new_q and new_q != question:
                        answered_count += 1
                        last_question = new_q
                        stale_rounds = 0  # Reset stale rounds on progress
                        print(f"    --> Save advanced to next question: '{new_q}'")
                    else:
                        stale_rounds += 1
                        print(f"    --> After Save, question did not advance (stale_rounds={stale_rounds}).")
                        # Allow more retries (4 instead of 2) - sometimes UI takes time to update
                        if stale_rounds >= 4:
                            print(f"    --> Question stalled after {stale_rounds} attempts. Checking if application might have succeeded...")
                            # Before giving up, check URL for success patterns
                            try:
                                current_url = getattr(page, "url", "") or ""
                                low_url = current_url.lower()
                                if "saveapply" in low_url or "/myapply/" in low_url or "myapply" in low_url:
                                    print(f"    --> Detected apply redirect URL: {current_url}; treating as submitted.")
                                    return True
                                page_text = await page.content()
                                if is_application_success_page(page_text, current_url):
                                    print("    --> Application success detected despite stale question.")
                                    return True
                            except Exception:
                                pass
                            
                            try:
                                os.makedirs("artifacts", exist_ok=True)
                                ts = int(time.time())
                                qsafe = "".join([c for c in question if c.isalnum() or c.isspace()])[:80].strip().replace(" ", "_")
                                fname_png = f"artifacts/failed_fill_no_advance_{ts}_{qsafe}.png"
                                try:
                                    await page.screenshot(path=fname_png, full_page=True)
                                    fname = fname_png
                                except Exception:
                                    try:
                                        html = await page.content()
                                        fname_html = f"artifacts/failed_fill_no_advance_{ts}_{qsafe}.html"
                                        with open(fname_html, "w", encoding="utf-8") as hf:
                                            hf.write(html)
                                        fname = fname_html
                                    except Exception:
                                        fname = None
                                with open("artifacts/failed_fills.log", "a", encoding="utf-8") as lf:
                                    lf.write(f"{datetime.now().isoformat()} no_advance_after_save question={question} url={getattr(page,'url','')} screenshot={fname}\n")
                                print(f"    --> Saved artifact for no-advance after Save: {fname}")
                            except Exception as e:
                                print(f"    --> Failed saving artifact for no-advance after Save: {e}")
                            return False

    # Final check after loop
    await page.wait_for_timeout(500)  # Reduced from 2000ms
    try:
        page_text = await page.content()
        current_url = getattr(page, "url", "")
        if is_application_success_page(page_text, current_url):
            print("--> Application submitted successfully after completing all questions!")
            return True
    except Exception:
        pass

    print(f"--> Questionnaire loop finished. Answered {answered_count} question(s).")
    return answered_count > 0


async def _handle_apply_flow(context, page, answers: list) -> bool:
    """Handle the post-Apply flow: chatbot drawer, new tabs, or inline pages.

    Returns True if the screening/questionnaire was processed successfully.
    """
    # Naukri-specific drawer selectors (chatbot-style questionnaire)
    drawer_selector = ", ".join([
        ".chatbot_drawer",
        ".chatbot_container",
        ".screeningQuestions",
        ".screening-questions",
        ".drawer-wrapper",
        "[class*='chatbot']",
        "[class*='screening']",
        ".apply-message-container",
        "[role='dialog']",
    ])

    try:
        # 1) Wait for the chatbot drawer/modal to appear on the same page
        try:
            await page.wait_for_selector(drawer_selector, timeout=5000)  # Reduced from 8000ms
            print("--> Screening drawer detected on current page.")
            submitted = await process_screening_modal(page, answers)
            if submitted:
                return True
        except Exception:
            pass

        # 2) Check if Apply opened a new page/tab
        try:
            new_page = await context.wait_for_event("page", timeout=5000)
            if new_page:
                await new_page.wait_for_load_state("domcontentloaded", timeout=10000)
                # Wait a bit for the chatbot to load on the new page - reduced from 3000ms
                await new_page.wait_for_timeout(1000)
                submitted = await process_screening_modal(new_page, answers)
                return bool(submitted)
        except Exception:
            pass

        # 3) Fallback: check if the page itself shows success (direct apply without screening)
        try:
            page_text = await page.content()
            current_url = getattr(page, "url", "")
            if is_application_success_page(page_text, current_url):
                print("--> Application submitted directly (no screening questions).")
                return True
        except Exception:
            pass

        # 4) Last resort: wait a bit more and try again
        try:
            await page.wait_for_timeout(2000)  # Reduced from 5000ms
            submitted = await process_screening_modal(page, answers)
            if submitted:
                return True
        except Exception:
            pass

        return False
    except Exception as e:
        print(f"Error during apply flow handling: {e}")
        return False


async def run():
    async with async_playwright() as p:
        # Launch browser keeping max 2 tabs total (Tab 1: Search Page, Tab 2: Job Details Page)
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./naukri_session",
            headless=False,
            args=["--start-maximized"]
        )
        
        # Tab 1: Primary Search Page
        search_page = context.pages[0] if context.pages else await context.new_page()
        print("Navigating to Naukri homepage...")
        await search_page.goto("https://www.naukri.com/", wait_until="domcontentloaded")
        await search_page.wait_for_timeout(2000)  # Reduced from 3000ms

        print("Navigating to Recommended section...")
        try:
            # Try to click on the recommended jobs link
            recommended_link = await search_page.wait_for_selector("a[href*='recommendedjobs']", timeout=5000)
            if recommended_link:
                await recommended_link.click()
                print("Clicked on Recommended section.")
            else:
                print("Navigating directly to recommended URL fallback.")
                await search_page.goto("https://www.naukri.com/mnj/recommendedjobs", wait_until="domcontentloaded")
            await search_page.wait_for_timeout(3000)  # Reduced from 5000ms
        except Exception as e:
            print(f"Error clicking recommended section, falling back to direct URL: {e}")
            await search_page.goto("https://www.naukri.com/mnj/recommendedjobs", wait_until="domcontentloaded")
            await search_page.wait_for_timeout(3000)  # Reduced from 5000ms

        # Tab 2: Dedicated Single Tab for processing individual job links
        job_page = await context.new_page()

        job_cards = await search_page.query_selector_all(".srp-jobtuple-wrapper, .jobTuple, .cust-job-tuple")
        print(f"Found {len(job_cards)} job cards on page.")

        scraped_jobs = []
        evaluations_list = []
        max_evaluations = CONFIG.get("max_applications_per_run", 1)  # Use config value (default 1)
        applied_count = 0
        print("\n--- Phase 1: Evaluating and Instant-Applying Jobs ---")
        for idx, card in enumerate(job_cards):
            if idx >= max_evaluations:
                break

            try:
                title_elem = await card.query_selector(".title")
                if not title_elem:
                    continue
                    
                print(f"\n[Evaluating Job #{idx + 1}]")
                async with context.expect_page() as new_page_info:
                    await title_elem.click()
                current_job_page = await new_page_info.value
                await current_job_page.wait_for_load_state("domcontentloaded")
                await current_job_page.wait_for_timeout(2000)
                
                # Wait for JD element to load
                try:
                    jd_elem = await current_job_page.wait_for_selector(".styles_JDSummary__xA23_, .job-desc, .dang-inner-html", timeout=5000)
                    jd_text = await jd_elem.inner_text() if jd_elem else ""
                except Exception:
                    jd_text = await current_job_page.inner_text("body") # Fallback to body text if selector times out

                title = ""
                title_elem_detail = await current_job_page.query_selector(".job-title, h1, .title")
                if title_elem_detail:
                    title = (await title_elem_detail.inner_text()).strip()

                # Try to extract company name from common Naukri selectors
                company = ""
                for comp_sel in [".companyName", ".company", ".cmp-name", ".name", ".companyName a", ".company a"]:
                    try:
                        comp_el = await current_job_page.query_selector(comp_sel)
                        if comp_el:
                            txt = (await comp_el.inner_text()) or ""
                            if txt and txt.strip():
                                company = txt.strip()
                                break
                    except Exception:
                        continue

                q_elems = await current_job_page.query_selector_all(".question-text, .q-title")
                questions = [(await q.inner_text()).strip() for q in q_elems if await q.inner_text()]

                job_payload = {
                    "title": title or f"Scraped Naukri Job #{idx + 1}",
                    "company": company or "",
                    "description": jd_text,
                    "skills": extract_skill_tokens(jd_text),
                }

                print(f"    -> Job scraped: Title='{job_payload['title']}' | Company='{job_payload.get('company','')}'")
                scraped_jobs.append(job_payload)

                # Rank this actual scraped job list before sending to Gemini.
                match_context = build_match_context(CONFIG["candidate"], scraped_jobs, top_n=min(5, len(scraped_jobs)))
                evaluation = evaluate_jd_and_questions_gemini(jd_text, questions, match_context=match_context)
                print(f"Match Score: {evaluation.get('match_score', 0)}% | Reason: {evaluation.get('reason')} | Job: '{job_payload['title']}' @ '{job_payload.get('company','')}'")
                if match_context.get("local_matcher", {}).get("status") == "ok":
                    print(f"Local matching context: {json.dumps(match_context, indent=2)}")

                evaluations_list.append({
                    "idx": idx,
                    "score": evaluation.get("match_score", 0),
                    "answers": evaluation.get("answers", []),
                    "job": job_payload,
                })

                # Instant-apply: if the model/agent recommends applying, click Apply and handle screening immediately.
                if evaluation.get("should_apply"):
                    print(f"--> Decision: apply to Job #{idx + 1} | Job: '{job_payload['title']}' @ '{job_payload.get('company','')}'")
                    apply_btn = await current_job_page.query_selector("button:has-text('Apply')")
                    if apply_btn:
                        btn_text = (await apply_btn.inner_text()).lower()
                        if "company site" in btn_text:
                            print("--> Redirects to external ATS. Skipping instant apply.")
                        else:
                            await apply_btn.click()
                            print("--> Clicked 'Apply'.")
                            await current_job_page.wait_for_timeout(1000)  # Reduced from 2000ms

                            # Ensure we have per-question answers; fall back to local reasoner if missing
                            answers = evaluation.get("answers") or ReasoningAgent().build_answers_for_questions(CONFIG["candidate"], questions)
                            submitted = await _handle_apply_flow(context, current_job_page, answers)
                            if submitted:
                                applied_count += 1
                                print("--> Application submitted; waiting up to 10s for confirmation message...")
                                # Wait up to 10 seconds for a clear confirmation marker on the page
                                confirmed = False
                                total_wait = 0
                                poll_interval = 1
                                max_wait = 10
                                for _wait in range(0, max_wait, poll_interval):
                                    await asyncio.sleep(poll_interval)
                                    total_wait += poll_interval
                                    try:
                                        page_text = await current_job_page.content()
                                        current_url = getattr(current_job_page, "url", "")
                                        if is_application_success_page(page_text, current_url):
                                            confirmed = True
                                            break
                                    except Exception:
                                        pass

                                    # Also check for visible UI text like 'Applied to' or 'Applied'
                                    try:
                                        el = await current_job_page.query_selector("text=Applied to")
                                        if not el:
                                            el = await current_job_page.query_selector("text=Applied")
                                        if el and await el.is_visible():
                                            confirmed = True
                                            break
                                    except Exception:
                                        pass

                                if confirmed:
                                    print("--> Application confirmed on page; closing successful job tab.")
                                else:
                                    print("--> No confirmation seen after 10s; closing tab to continue.")
                                try:
                                    await current_job_page.close()
                                except Exception:
                                    pass
                                await asyncio.sleep(2.0)
                                continue
                            else:
                                # save artifact for manual inspection
                                try:
                                    os.makedirs("artifacts", exist_ok=True)
                                    ts = int(time.time())
                                    fname = f"artifacts/unanswered_job_{idx+1}_{ts}.png"
                                    try:
                                        await current_job_page.screenshot(path=fname, full_page=True)
                                    except Exception:
                                        # page may be closed; try to save page HTML fragment instead
                                        try:
                                            html = await current_job_page.content()
                                            hname = f"artifacts/unanswered_job_{idx+1}_{ts}.html"
                                            with open(hname, "w", encoding="utf-8") as hf:
                                                hf.write(html)
                                            fname = hname
                                        except Exception:
                                            fname = None
                                    with open("artifacts/unanswered_jobs.log", "a", encoding="utf-8") as lf:
                                        lf.write(f"{datetime.now().isoformat()} idx={idx+1} title={title} company={company} url={current_job_page.url} screenshot={fname}\n")
                                    print(f"--> Saved screenshot and log for unanswered job: {fname} (title={title} company={company})")
                                except Exception as e:
                                    print(f"--> Failed saving artifact: {e}")
                    else:
                        print("--> Apply button missing or already applied.")

                if not current_job_page.is_closed():
                    try:
                        await current_job_page.close()
                    except Exception:
                        pass
                await asyncio.sleep(20.0)  # Updated delay to respect free tier quota (approx 3 requests/min)
            except Exception as e:
                print(f"Error evaluating job #{idx + 1}: {e}")
                if 'current_job_page' in locals() and not current_job_page.is_closed():
                    await current_job_page.close()
                    
        print(f"\nInstant-apply phase complete. Total applied_count: {applied_count}")

        # Close worker tab when done
        await job_page.close()
        print("\nPipeline execution complete.")

if __name__ == "__main__":
    # Number of full pipeline cycles to execute. 2 evaluations per cycle => 20 requests/day.
    max_cycles = 1
    async def repeat_runs():
        for cycle in range(1, max_cycles + 1):
            print(f"\n=== Cycle {cycle} of {max_cycles} ===")
            try:
                await run()
            except Exception as e:
                # Basic handling for rate limit errors – wait and continue.
                print(f"Run failed with {e}, sleeping 30s before next cycle.")
                await asyncio.sleep(30)
            if cycle < max_cycles:
                # Wait before next cycle to stay within quota limits.
                await asyncio.sleep(60)  # 1 minute pause between full runs
        print("\nAll cycles completed.")
    asyncio.run(repeat_runs())