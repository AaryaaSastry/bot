from config import call_gemini


def ask_question(state):
    """Agent 1: Professional Medical Interviewer"""

    # Phase 1: If diagnosis was rejected, ask clarification yes/no questions
    if getattr(state, "extra_questions_asked", False):

        prompt = f"""
You are Agent 1, a skilled, mindful,  ayurvedic medical interviewer.

A previous diagnosis was rejected and you must clarify symptoms.

Conversation History:
{state.get_conversation_history()}

Ask ONE clear YES/NO medical question that helps narrow the diagnosis.
Ensure the question is relevant to the symptoms and has not been asked before.
Ensure the questions dont have any ambigity and can be answered with a clear yes or no. Avoid asking multiple questions at once. Focus on one symptom or risk factor at a time.
If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"

Return ONLY the question.
"""
        return call_gemini(prompt).strip()

    # First turn → greeting
    if state.turn_count == 0:
        return "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"

    # Second turn → determine if user provided symptoms
    if state.turn_count == 1:

        prompt = f"""
You are Agent 1 (Medical Interviewer).

User response:
{state.answers[-1]}

Conversation History:
{state.get_conversation_history()}

If the user only greeted you (like "hi", "hello"), politely ask them to describe their medical symptoms.

If they mentioned symptoms, ask ONE relevant follow-up question to clarify them.
Ensure the question is relevant to the symptoms and has not been asked before.
Ensure the questions dont have any ambigity and can be answered with a clear yes or no. Avoid asking multiple questions at once. Focus on one symptom or risk factor at a time.
If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"
Return ONLY the question.
"""
        return call_gemini(prompt).strip()

    # Later turns → follow-up questions
    feedback_context = ""
    feedbacks = getattr(state, "verification_feedbacks", [])

    if feedbacks:
        latest = feedbacks[-1]
        feedback_context = f"Supervisor feedback from Agent 2:\n{latest}"

    prompt = f"""
You are Agent 1, a medical interviewer gathering diagnostic information.

Known symptoms so far:
{state.symptoms_positive}

Conversation History:
{state.get_conversation_history()}

{feedback_context}

Your task:
- Ask ONE new medical question that has NOT already been asked.
- Avoid repeating previous questions.
- Focus on clarifying symptoms or identifying related symptoms.

Return ONLY the question text.
"""

    return call_gemini(prompt).strip()


def generate_simple_diagnosis(state):
    """Agent 1: Preliminary diagnostic reasoning"""

    prompt = f"""
You are Agent 1, performing a preliminary medical analysis.
Your task is to understand the User properly like an experienced ayurvedic doctor and generate a simple diagnostic hypothesis based on the symptoms identified so far.
You need to ensure u take decent analysis of the symptoms and their patterns, and not just list them.

Known symptoms:
{state.symptoms_positive}

Conversation History:
{state.get_conversation_history()}

Generate a simple diagnostic hypothesis CONSIDERING the patterns of symptoms, their severity, and any relevant risk factors.
Focus on the most likely condition that could explain the symptoms, but you can also mention a few other possibilities if they are relevant.
Keep the diagnosis concise and focused on the most likely condition, but you can also mention a few other possibilities if they are relevant.


Output format:
POSSIBLE CONDITION:
Name of condition.

REASONING:
Explain how the symptoms support this condition.

Return only this analysis.
"""

    return call_gemini(prompt)