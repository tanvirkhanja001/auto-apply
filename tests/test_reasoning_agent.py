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
    assert QuestionnaireAgent.normalize_field_answer("text", "30 days") == "30 days"


def test_application_success_page_is_detected_only_for_real_success_states():
    assert is_application_success_page("Your application has been submitted successfully.") is True
    assert is_application_success_page("Save", "https://naukri.com/apply") is False
