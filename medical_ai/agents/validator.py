from config import call_llm, SYSTEM_PROMPTS


def verify_questions(state):
    """
    Agent 2 (Supervisor):
    Reviews Agent 1's questioning strategy and suggests improvements.
    """

    history = state.get_conversation_history()

    prompt = f"""
You are Agent 2, a highly experienced Senior Ayurvedic Medical Supervisor reviewing a junior doctor's interview.

Your goal is to ensure the clinical reasoning is deep, accurate, and follows classical Ayurvedic diagnostic principles (Trividha Pariksha: Darshana, Sparshana, Prashna).

Tasks:
1. Evaluate if Agent 1's questions are medically useful for an Ayurvedic diagnosis.
2. Identify missing diagnostic signals (e.g., Agni status, Ama presence, Guna/Dosha patterns).
3. Check for any clinical red flags or severity levels that been missed or downplayed.
4. Suggest sophisticated questions that probe deeper into the root cause (Hetu) rather than just symptoms.

Conversation History:
{history}

Evaluate the questioning strategy using the following format:

STRENGTHS:
- Concise points on what Agent 1 did well.

MISSING INFORMATION:
- Crucial Ayurvedic indicators missing (e.g., digestion patterns, sleep quality, tongue/stool characteristics if relevant).

CLINICAL CONCERNS:
- Any risks or symptoms requiring more urgent focus.

SUGGESTED QUESTIONS:
- Provide 3–5 sophisticated follow-up question ideas. Do not provide the exact wording, but the objective of the questions.

Return ONLY this structured evaluation.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["validator"])


def create_final_summary(state):
    """
    Agent 2:
    Generates a comprehensive clinical report based on Ayurvedic principles.
    """

    history = state.get_conversation_history()

    prompt = f"""
You are Agent 2, a Senior Ayurvedic Clinical Reviewer.

Produce a detailed, professional Ayurvedic clinical report based on the interaction.

Symptoms Identified:
{state.symptoms_positive}

Agent 1 Reasoning:
{state.agent1_summary}

Conversation History:
{history}

Create the final report using this structure:

CLINICAL DYNAMICS (Samprapti):
- Describe the Dosha imbalance (Vata/Pitta/Kapha).
- Describe the state of Agni (Mandagni, Vishamagni, etc.) and presence of Ama.
- Explain how the symptoms relate to these imbalances.

PROBABLE AYURVEDIC CONDITIONS (Roga Vinishchaya):
- Primary condition and 2 differential diagnoses with brief Ayurvedic justification.

SEVERITY & PROGNOSIS (Sadhya-Asadhyata):
- LEVEL: Low / Moderate / High
- Explanation based on symptom duration and intensity.

RECOMMENDED GUIDELINES:
- Immediate lifestyle or dietary suggestions (Ahar/Vihar) aligned with the suspected imbalance.
- Advice on seeking professional Ayurvedic consultation.

Return ONLY the structured report.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["validator"])