import importlib.util

from agents.questionnaire_agent import QuestionnaireAgent
from agents.reasoning_agent import ReasoningAgent

spec = importlib.util.spec_from_file_location("main_gemini", "main-gemini.py")
main_gemini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_gemini)
is_application_success_page = main_gemini.is_application_success_page


def test_relocation_question_uses_yes_no_answer():
    candidate = {
        "name": "Tanvir Jahagirdar",
        "current_location": "Nashik",
        "preferred_locations": ["Pune", "Bangalore"],
        "notice_period_days": "30",
        "expected_ctc_lpa": "18",
        "total_exp_years": "3",
    }
    question = "Are you currently living in or ready to relocate to Bengaluru?"

    answer = ReasoningAgent().answer_question(candidate, question)

    assert answer.strip().lower() in {"yes", "yes, i am open to relocating", "yes, i can relocate", "available to relocate"}


def test_radio_answers_are_normalized_for_questionnaire_fields():
    assert QuestionnaireAgent.normalize_field_answer("radio", "Yes") == "yes"
    assert QuestionnaireAgent.normalize_field_answer("radio", "No") == "no"
    assert QuestionnaireAgent.normalize_field_answer("checkbox", "Yes") == "yes"
    assert QuestionnaireAgent.normalize_field_answer("checkbox", "No") == "no"
    assert QuestionnaireAgent.normalize_field_answer("text", "30 days") == "30 days"


def test_compound_relocation_question_prefers_yes_over_notice_period_heuristic():
    candidate = {
        "name": "Tanvir Jahagirdar",
        "current_location": "Nashik",
        "preferred_locations": ["Pune", "Bangalore"],
        "notice_period_days": "30",
        "expected_ctc_lpa": "18",
        "total_exp_years": "3",
    }
    question = "Are you willing to relocate to Pune or Bangalore and available to join in 30 days?"

    answer = ReasoningAgent().answer_question(candidate, question)

    assert answer.strip().lower() in {"yes", "yes, i am open to relocating", "yes, i can relocate", "available to relocate"}


def test_question_answer_uses_available_options_for_checkbox_and_radio_choices():
    candidate = {
        "name": "Tanvir Jahagirdar",
        "primary_skills": ["Java", "Spring Boot", "React.js", "AWS"],
        "preferred_locations": ["Pune", "Bangalore"],
        "total_exp_years": "3",
    }
    agent = ReasoningAgent()

    skill_options = ["Java", "Spring Boot", "React.js", "AWS", "Node.js"]
    selected_skills = agent.match_answer_to_options(
        candidate,
        "Which of these technologies do you have hands-on experience with?",
        "I have hands-on experience with Java, Spring Boot, React.js and AWS.",
        skill_options,
    )
    assert set(["Java", "Spring Boot", "React.js", "AWS"]).issubset(set([selected_skills])) or selected_skills in skill_options

    relocation_choice = agent.match_answer_to_options(
        candidate,
        "Are you willing to relocate?",
        "Yes",
        ["Yes", "No"],
    )
    assert relocation_choice.lower() == "yes"


def test_gemini_fallback_is_used_when_ollama_is_unavailable(monkeypatch):
    agent = ReasoningAgent()
    agent.remote_available = False
    monkeypatch.setattr(agent, "_call_ollama", lambda *args, **kwargs: "")
    monkeypatch.setattr(agent, "_call_gemini", lambda *args, **kwargs: "Yes")

    answer = agent.answer_question({"current_location": "Nashik", "preferred_locations": ["Pune", "Bangalore"]}, "Are you willing to relocate to Pune or Bangalore?")

    assert answer == "Yes"


def test_gemini_uses_supported_model_names(monkeypatch):
    agent = ReasoningAgent()
    seen = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            seen.append(model)
            raise RuntimeError("simulated failure")

    class FakeClient:
        models = FakeModels()

    agent.gemini_client = FakeClient()
    assert agent._call_gemini("hello") == ""
    assert seen[0] == "gemini-3.5-flash"
    assert all(model in {"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"} for model in seen)
    assert "gemini-1.5-flash" not in seen
    assert "gemini-2.5-flash" not in seen


def test_reasoning_cache_reuses_same_question_answer():
    candidate = {
        "current_location": "Nashik",
        "preferred_locations": ["Pune", "Bangalore"],
        "notice_period_days": "30",
        "expected_ctc_lpa": "18",
        "total_exp_years": "3",
    }
    agent = ReasoningAgent()
    calls = {"count": 0}

    def fake_ollama(prompt, timeout=3, model_name=None, **kwargs):
        calls["count"] += 1
        return "Yes"

    agent._call_ollama = fake_ollama
    agent._call_gemini = lambda *args, **kwargs: ""

    first = agent.answer_question(candidate, "Are you willing to relocate to Pune?", ["Yes", "No"])
    second = agent.answer_question(candidate, "Are you willing to relocate to Pune?", ["Yes", "No"])

    assert first == "Yes"
    assert second == "Yes"
    assert calls["count"] == 1


def test_application_success_page_is_detected_only_for_real_success_states():
    assert is_application_success_page("Your application has been submitted successfully.") is True
    assert is_application_success_page("Save", "https://naukri.com/apply") is False
