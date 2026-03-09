from config import call_gemini


def verify_questions(state):
    """
    Agent 2 (Supervisor):
    Reviews Agent 1's questioning strategy and suggests improvements.
    """

    history = state.get_conversation_history()

    prompt = f"""
You are Agent 2, a senior medical supervisor reviewing a junior doctor's interview.

Your task is to evaluate whether Agent 1's questions are medically useful
and if important symptoms are missing.

Conversation History:
{history}

Evaluate the questioning strategy using the following format.

STRENGTHS:
- What Agent 1 did well.

MISSING INFORMATION:
- Important symptoms or diagnostic questions that were not asked.

CLINICAL CONCERNS:
- Any dangerous symptoms that should be prioritized.

SUGGESTED QUESTIONS:
- Provide 3–5 better follow-up question ideas that Agent 1 should ask next. DO NOT DIRECTLY PRESENT THE QUESTION. 

Return ONLY this structured evaluation.
"""

    return call_gemini(prompt)


def create_final_summary(state):
    """
    Agent 2:
    Generates the final clinical report and verifies Agent 1's diagnosis.
    """

    history = state.get_conversation_history()

    prompt = f"""
You are Agent 2, a medical reviewer.

Review the interaction and produce a structured clinical report.

Symptoms Identified:
{state.symptoms_positive}

Agent 1 Diagnosis:
{state.agent1_summary}

Conversation History:
{history}

Create the final report using this structure:

CLINICAL SUMMARY:
Summarize the key symptoms and their progression.

POSSIBLE CONDITIONS:
List 3 possible conditions with short reasoning for each.

DANGER LEVEL:
Low / Moderate / High with explanation.

RECOMMENDED NEXT STEPS:
What the patient should do next (medical consultation, emergency care, etc).

Return ONLY the structured report.
"""

    return call_gemini(prompt)