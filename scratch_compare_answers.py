import json
from agents.reasoning_agent import ReasoningAgent

with open('config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

candidate = cfg['candidate']
agent = ReasoningAgent()
questions = [
    'Are you willing to relocate?',
    'How many years of experience do you have?',
    'What is your expected CTC?',
    'Are you currently living in or ready to relocate to Bengaluru?',
    'How many days notice period do you have?',
    'Are you willing to relocate to Pune or Bangalore and available to join in 30 days?',
    'Do you have hands-on experience with Java, Spring Boot, AWS, and React.js for backend and frontend roles?',
    'What is your current CTC and what is your expected CTC in LPA?',
    'Are you currently living in Nashik and are you open to relocate to Pune or Mumbai?',
    'How many years of experience do you have and what is your notice period?'
]

for q in questions:
    print(f'Q: {q}')
    print(f'A: {agent.answer_question(candidate, q)}')
    print()

options = ['0-1 years', '1-3 years', '3-5 years', '5-8 years', '8+ years']
print('Match test:', agent.match_answer_to_options(candidate, 'How many years of experience do you have?', '3 years', options))
