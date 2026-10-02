from typing import Dict, List
from agents.reasoning_agent import ReasoningAgent

class QuestionnaireAgent:
    """Detect application questions and answer using resume/profile and approved answers."""

    def __init__(self, candidate: Dict, model_name: str = 'qwen2.5:0.5b-instruct', ollama_model_order: List[str] | None = None):
        self.candidate = candidate
        self.reasoner = ReasoningAgent(
            model_name=model_name,
            ollama_model_order=ollama_model_order or [
                'gemini-3.5-flash',
                model_name,
                'SmolLM2-360M-Instruct',
            ],
        )

    @staticmethod
    def normalize_field_answer(field_type: str, answer: str) -> str:
        """Normalize field values for radio/check/select questions in the live ATS form."""
        value = (answer or "").strip()
        if not value:
            return ""

        lowered = value.lower()
        field_type = (field_type or "").lower()

        if field_type in {"radio", "checkbox"}:
            if lowered in {"y", "yes", "yes please", "available", "open to relocate", "ready to relocate", "willing to relocate", "i am open to relocating", "available to join immediately", "i can relocate"}:
                return "yes"
            if lowered in {"n", "no", "not available", "not willing", "not open to relocate", "not ready to relocate", "cannot relocate"}:
                return "no"

        return lowered

    def detect_and_answer(self, questions: List[str]) -> List[Dict]:
        results = []
        for q in questions:
            try:
                ans = self.reasoner.answer_question(self.candidate, q)
            except Exception as ex:
                # Log the error and fall back to a safe default answer detection
                print(f"QuestionnaireAgent: reasoner failed for question '{q}': {ex}")
                ans = ""
            # If answer contains unknown tokens, mark as needs user
            if ans.lower().strip() in ['', 'unknown', 'n/a', 'i don\'t know']:
                print(f"QuestionnaireAgent: marking question as NEEDS_USER: '{q}' (ans='{ans}')")
                results.append({'question': q, 'status': 'NEEDS_USER', 'answer': None})
            else:
                print(f"QuestionnaireAgent: generated answer for question: '{q}' -> '{ans}'")
                results.append({'question': q, 'status': 'ANSWER', 'answer': ans})
        return results
