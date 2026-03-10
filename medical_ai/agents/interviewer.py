
from config import call_llm, SYSTEM_PROMPTS

                
def is_health_related(user_input):
    """Check if user input is related to health/medical symptoms"""
    
    prompt = f"""
You are a medical domain classifier. Your task is to determine if the user's input is related to health, medical symptoms, or Ayurvedic diagnostics.

User input: "{user_input}"

Determine if this input is:
- HEALTH-RELATED: The user is describing symptoms, medical conditions, body issues, pain, discomfort, or asking health-related questions
- NOT HEALTH-RELATED: The user is asking about unrelated topics like objects, general knowledge, math, weather, animals (unless medically relevant), hobbies, etc.

Examples of NOT HEALTH-RELATED inputs:
- "What is a pencil?" - asking about an object
- "What's the weather today?" - asking about weather
- "Tell me about history" - asking about general knowledge
- "How do I fix my computer?" - technical question
- "What is 2+2?" - math question
- "Tell me about cars" - general topic

Examples of HEALTH-RELATED inputs:
- "I have a headache" - describing symptom
- "My stomach hurts" - describing symptom
- "I feel tired all the time" - describing symptom
- "What causes fever?" - medical question
- "I have back pain" - describing symptom

Respond with ONLY one word: HEALTHY or NOT_HEALTHY
"""
    
    result = call_llm(prompt, SYSTEM_PROMPTS.get("symptom_extractor", ""), temperature=0.3).strip().upper()
    return "HEALTHY" in result


def get_domain_redirect_response():
    """Return a polite response asking user to discuss only health problems"""
    return "I appreciate your question, but I'm specifically designed to help with health-related concerns and medical symptoms. Please tell me about any health problems or symptoms you're experiencing, and I'll be happy to assist you from an Ayurvedic perspective."


def ask_question(state):
    """Agent 1: Professional Medical Interviewer"""

    # Phase 1: If diagnosis was rejected, ask clarification yes/no questions
    if getattr(state, "extra_questions_asked", False):
        
        # Check if user input is health-related
        user_input = state.answers[-1] if state.answers else ""
        if user_input and not is_health_related(user_input):
            return get_domain_redirect_response()

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
        
        # Check if user input is health-related
        user_input = state.answers[-1] if state.answers else ""
        if user_input and not is_health_related(user_input):
            return get_domain_redirect_response()

        prompt = f"""
You are Agent 1 (Medical Interviewer).

User response:
{state.answers[-1]}

Conversation History:
{state.get_conversation_history()}
BASE RULE:
    - You are an ayurvedic medical interviewer, not a general chatbot. Always steer the conversation towards understanding the user's medical symptoms and concerns.
    - You need to understand the user properly like an experienced ayurvedic doctor, and not just list the symptoms. You need to take decent analysis of the symptoms and their patterns, and not just list them.
    - If the user asks about specific medications like paracetamol or any non-ayurvedic treatments, inform them that you can't advise on that as it is outside your ayurvedic expertise. Redirect them to describe their symptoms so you can help them from an ayurvedic perspective.

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
    # Check if user input is health-related
    user_input = state.answers[-1] if state.answers else ""
    if user_input and not is_health_related(user_input):
        return get_domain_redirect_response()

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
- If the user has asked about non-Ayurvedic medications or treatments (for example, Paracetamol), politely inform them that you are an Ayurvedic diagnostic assistant and can only help with symptoms, then redirect the conversation to their symptoms.
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