# auto-apply

Automated job application bot for Naukri.com using Playwright and AI-powered screening question handling.

## Features

- **Resume-Based Job Matching**: Evaluates job fit against candidate profile (skills, experience, location, notice period, CTC) using local embeddings (BAAI/bge-small-en-v1.5) or Gemini API
- **Automated Job Discovery**: Scans Naukri recommended jobs and applies to matching positions
- **Screening Questionnaire Handler**: Handles Naukri's chatbot-style screening questions (radio buttons, text inputs, dropdowns)
- **AI-Powered Answer Matching**: Uses ReasoningAgent to match candidate answers to radio/select options
- **Configurable Limits**: Set max applications per run via config
- **Artifact Collection**: Screenshots/HTML saved for failed fills for debugging

## Configuration

### config.json

Edit `config.json` with your candidate profile:

```json
{
  "search_url": "https://www.naukri.com/java-developer-jobs-in-pune",
  "max_applications_per_run": 5,
  "candidate": {
    "name": "Your Name",
    "email": "your@email.com",
    "phone": "9876543210",
    "total_exp_years": 3,
    "notice_period_days": 30,
    "expected_ctc_lpa": 18,
    "current_location": "Pune",
    "primary_skills": ["Java", "Spring Boot", "Microservices"],
    "secondary_skills": ["AWS", "Docker", "React"]
  }
}
```

**Job Matching Criteria:**
- **Skills**: Primary + secondary skills matched against job requirements (cosine similarity via embeddings)
- **Experience**: Total years matched against job's required range
- **Location**: Current location vs job location (hybrid/remote considered)
- **Notice Period**: Matched against employer preference
- **CTC**: Expected vs offered range

### .env.local

Create `.env.local` for API keys and secrets:

```bash
# Gemini API (optional - falls back to local embeddings if not set)
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama URL for local reasoning agent (default: http://localhost:11434)
OLLAMA_URL=http://localhost:11434

# Naukri session persistence (auto-created)
NAUKRI_SESSION_DIR=./naukri_session
```

**Note**: Never commit `.env.local` to version control. Add to `.gitignore`.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run
python main-gemini.py
```

## Key Fixes Implemented

### 1. Save Button Detection (`_click_save_in_drawer`)
- Added Naukri-specific selectors: `button[id*='sendMsgbtn']`, `button[id*='sendMsgbtn_container']`
- Reduced wait time from 5000ms to 2000ms per selector
- Added "already saved" state detection (button text "Saved")
- JS click fallback for viewport issues

### 2. Radio Button Matching
- Fixed radio grouping logic (`radios_by_name` now properly populated)
- Added `ReasoningAgent.match_answer_to_options()` for AI-powered option matching
- Local fallback matching for experience ranges (e.g., "3 years" → "3-4 years")
- JS click fallback when regular click fails (element outside viewport)

### 3. Drawer Close Handling
- Immediate success detection on drawer close
- Checks for Naukri redirect URLs (`saveapply`, `myapply`)
- Checks for "Thank you for your responses" completion message
- Increased stale tolerance from 3 to 5 rounds

### 4. Performance Optimizations
- Reduced all `wait_for_timeout` calls (3000→1500, 2000→1000, 1500→800, etc.)
- Config-driven `max_applications_per_run` limit
- Faster drawer detection timeout (8000ms → 5000ms)

## Project Structure

```
auto-apply/
├── main-gemini.py          # Main entry point with all fixes
├── config.json             # Configuration
├── .env.local              # API keys (not committed)
├── agents/
│   ├── reasoning_agent.py  # AI reasoning for answer matching
│   ├── questionnaire_agent.py
│   ├── application_agent.py
│   └── orchestrator.py
├── artifacts/              # Failed fill screenshots/logs
└── naukri_session/         # Playwright persistent session
```

## Screening Flow

1. Click "Apply" on job card
2. Detect chatbot drawer (`.chatbot_Drawer`, `[class*='chatbot']`)
3. Read question from `.botMsg` bubbles
4. Get answer from pre-computed lookup or ReasoningAgent
5. Fill input:
   - **Radio**: Match answer to options via ReasoningAgent, click via JS
   - **Text/Number**: Fill contenteditable/textbox
   - **Select**: Select option by label/value
6. Click Save button (`sendMsgbtn` ID pattern)
7. Verify question advanced or completion detected
8. Repeat until "Thank you for your responses" or redirect

## Logs & Debugging

- `artifacts/failed_fills.log` - Failed question fills
- `artifacts/failed_fill_<timestamp>_<question>.png` - Screenshots
- `artifacts/unanswered_jobs.log` - Jobs with unanswered questions
- Console output shows all matching/clicking decisions