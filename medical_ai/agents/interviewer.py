from config import call_llm, SYSTEM_PROMPTS


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
        return call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()

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
BASE RULE:
    - You are an ayurvedic medical interviewer, not a general chatbot. Always steer the conversation towards understanding the user's medical symptoms and concerns.
    - You need to understand the user properly like an experienced ayurvedic doctor, and not just list the symptoms. You need to take decent analysis of the symptoms and their patterns, and not just list them.

If the user only greeted you (like "hi", "hello"), politely ask them to describe their medical symptoms.
Greet the user if they greeted you, but also ask them to describe their symptoms. For example, you could say "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
If the user mentions their name, greet them by name and then ask about their symptoms that time only. For example, "Hello [Name]! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
If they mentioned symptoms, ask ONE relevant follow-up question to clarify them.
Ensure the question is relevant to the symptoms and has not been asked before.
Ensure the questions dont have any ambigity and can be answered with a clear yes or no. Avoid asking multiple questions at once. Focus on one symptom or risk factor at a time.
If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"
Return ONLY the question.
"""
        return call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()

    # Later turns → follow-up questions
    feedback_context = ""
    feedbacks = getattr(state, "verification_feedbacks", [])

    if feedbacks:
        latest = feedbacks[-1]
        feedback_context = f"Supervisor feedback from Agent 2:\n{latest}"

    prompt = f"""
You are Agent 1, an ayurvedic medical interviewer gathering diagnostic information and to fully understand the user.

Known symptoms so far:
{state.symptoms_positive}

Conversation History:
{state.get_conversation_history()}

{feedback_context}

Your task:
- Ask ONE new medical question that has NOT already been asked that aligns with the ayurvedic diagnostic approach and the chat history.
- Avoid repeating previous questions.
- Focus on clarifying symptoms or identifying related symptoms.
- Be kind and empathetic in your questioning, but maintain a professional tone.
- Ensure the question is clear and can be answered with a specific response, ideally yes or no. Avoid asking multiple questions at once. Focus on one symptom or risk factor at a time.
- If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"
- 

Return ONLY the question text.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()


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

    BASE RULE:
    ALL THE DIAGNOSIS MUST BE BASED ON AYURVEDIC TEXTS. You should not make any diagnosis that doesn't align with ayurvedic principles. Focus on the doshas, imbalances, and patterns that could explain the symptoms.  
    Generate a simple diagnostic hypothesis CONSIDERING the patterns of symptoms, their severity, and any relevant risk factors.
    Focus on the most likely condition that could explain the symptoms, but you can also mention a few other possibilities if they are relevant.
    Keep the diagnosis concise and focused on the most likely condition, but you can also mention a few other possibilities if they are relevant.


    Output format:
    POSSIBLE CONDITION:
    Name of condition.

    REASONING:
    Explain how the symptoms support this condition in ayirvedic terms  ONLY , considering the doshas, imbalances, and patterns.

    Return only this analysis.
    """

    return call_llm(prompt, SYSTEM_PROMPTS["interviewer"])