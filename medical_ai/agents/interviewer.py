
from config import call_llm, SYSTEM_PROMPTS
from medical_ai.medical_framework import (
    RedFlagDetector, SymptomScorer, ProbabilisticDiagnosis,
    ConfidenceCalculator, DiagnosisRefusal, TwoPhaseSystem
)
from medical_ai.memory.diagnostic_state import DiagnosticState

                
def is_health_related(user_input):
    """Check if user input is related to health/medical symptoms"""
    
    prompt = f"""
You are an expert in medical domain classification. Your task is to determine if the user's input is related to health, medical symptoms, or Ayurvedic diagnostics.
Keep all the questions very small and precise. DO NOT ELABORATE THE QUESTIONS.

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
    return "I appreciate your question, but I'm specifically designed to help with health-related concerns and medical symptoms."


def ask_question(state):
    """Agent 1: Professional Ayurvedic Medical Interviewer"""

    # Phase 1: If diagnosis was rejected, ask clarification yes/no questions
    if getattr(state, "extra_questions_asked", False):
        
        # Check if user input is health-related
        user_input = state.answers[-1] if state.answers else ""
        if user_input and not is_health_related(user_input):
            return get_domain_redirect_response()

        prompt = f"""
You are Agent 1, a skilled Ayurvedic medical interviewer.

CRITICAL: Keep questions VERY SHORT (max 10 words).
A previous diagnosis was rejected and you must clarify symptoms.

Conversation History:
{state.get_conversation_history()}

Ask ONE clear YES/NO question (SHORT).
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
You are Agent 1 (Ayurvedic Medical Interviewer).

CRITICAL: Keep questions VERY SHORT (max 10 words).

User response:
{state.answers[-1]}

Conversation History:
{state.get_conversation_history()}

BASE RULE:
- Keep questions short and simple
- Ask one question at a time
- If user greeted you, greet back and ask about symptoms
- If they mentioned symptoms, ask ONE short follow-up

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
You are Agent 1, an ayurvedic medical interviewer gathering diagnostic information.

CRITICAL RULES:
1. Keep questions VERY SHORT (max 15-18 words)
2. Avoid medical jargon
3. Ask one question at a time only
4. Mix yes/no with occasional open questions

Known symptoms so far:
{state.symptoms_positive}

Conversation History:
{state.get_conversation_history()}

{feedback_context}

Your task:
- Ask ONE new question (SHORT, max 15-18 words)
- Never repeat previous questions
- Focus on: location, severity, duration, triggers
- 20% of the time, ask an open question like "Can you describe more?"

Return ONLY the question.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()


def generate_simple_diagnosis(state):
    """Agent 1: Preliminary diagnostic reasoning with probabilistic diagnosis"""

    # Calculate dosha scores
    dosha_scores = SymptomScorer.calculate_dosha_scores(state.symptoms_positive)
    dominant_dosha = SymptomScorer.get_dominant_dosha(dosha_scores)
    
    # Generate differential diagnoses
    differential = ProbabilisticDiagnosis.generate_differential_diagnosis(state.symptoms_positive)
    
    # Calculate data completeness
    symptom_category = TwoPhaseSystem.get_symptom_category(state.symptoms_positive)
    required_fields = TwoPhaseSystem.MINIMUM_FIELDS.get(symptom_category, ["onset", "severity"])
    collected_fields = {}
    # Estimate from conversation
    data_completeness = min(len(state.symptoms_positive) / len(required_fields), 1.0) if required_fields else 0
    
    # Include OPQRST data in the prompt
    opqrst_context = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""
    
    # Check if we have enough data for a realistic diagnosis
    if data_completeness < 0.6 or len(state.symptoms_positive) < 2:
        print("\n⚠️ Agent 1: I need more data to diagnose realistically. Continuing with more questions...\n")
    
    prompt = f"""
You are Agent 1, performing a preliminary medical analysis.

IMPORTANT: 
- Generate DIFFERENTIAL DIAGNOSES (multiple possibilities with probabilities)
- Calculate HONEST confidence based on information available.

Known symptoms:
{state.symptoms_positive}

{opqrst_context}

Dosha Analysis (Vata/Pitta/Kapha):
- Vata score: {dosha_scores.get('vata', 0)}
- Pitta score: {dosha_scores.get('pitta', 0)}
- Kapha score: {dosha_scores.get('kapha', 0)}
- Dominant: {dominant_dosha}

Data Completeness: {int(data_completeness * 100)}%

Conversation History:
{state.get_conversation_history()}

BASE RULE:
ALL DIAGNOSIS MUST BE BASED ON AYURVEDIC PRINCIPLES.

Output format:
DIFFERENTIAL DIAGNOSES:
1. [Condition name] - [probability]% - [brief reasoning]
2. [Condition name] - [probability]% - [brief reasoning]
3. [Condition name] - [probability]% - [brief reasoning]

DOSHA IMBALANCE:
Explain the dominant dosha pattern

CONFIDENCE SCORE:
Provide a score (0-1) - MUST BE LOW if data is incomplete

Return only this analysis.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["interviewer"])


def ask_opqrst_question(state):
    """Ask OPQRST questions - must collect all 6 fields before proceeding"""
    
    # Get all OPQRST fields from DiagnosticState
    opqrst_fields = DiagnosticState.OPQRST_FIELDS
    
    # Find first missing field
    missing_fields = state.get_missing_opqrst_fields()
    
    if not missing_fields:
        # All OPQRST fields collected
        return None
    
    current_field = missing_fields[0]
    
    # OPQRST questions mapping
    opqrst_questions = {
        "onset": "When did this symptom start? (e.g., today, yesterday, a week ago)",
        "provocation": "What makes it worse or better?",
        "quality": "How would you describe the pain? (throbbing, sharp, dull, burning)",
        "region": "Where exactly is the symptom located?",
        "severity": "How severe is it on a scale of 1-10?",
        "time": "How long does it last? / How often does it occur?"
    }
    
    question = opqrst_questions.get(current_field, f"Please tell me about {current_field}")
    
    return {
        "question": question,
        "field": current_field
    }