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
        user_input = state.answers[-1] if state.answers else ""
        if user_input and not is_health_related(user_input):
            return get_domain_redirect_response()

        prompt = f"""
You are Agent 1, a professional Ayurvedic medical interviewer.

GOAL:
Collect accurate clinical information needed for diagnosis.

REASONING INSTRUCTION:
Analyze the symptoms, OPQRST information, and conversation history carefully.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

CRITICAL RULE:
Use ONLY the structured data below.
Do NOT extract symptoms from conversation history.
Do NOT reinterpret answers.

Structured Data:
Onset: {state.collected_fields.get('onset','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Severity: {state.collected_fields.get('severity','unknown')}
Time: {state.collected_fields.get('time','unknown')}

Additional Symptoms:
{state.symptoms_positive}

RULES:
- Ask only ONE question at a time
- Avoid repeating questions
- Prefer short clear questions
- Focus on missing diagnostic information
- Use a mix of YES/NO and short open questions
STRICT CLINICAL RULES:
1. Never invent symptoms
2. If the patient denies a symptom, exclude it
3. Use only explicitly stated symptoms
4. Do not reinterpret answers
5. Use ONLY the structured data below; do NOT extract new info from conversation history.

PRIORITY INFORMATION:
1. Location of symptoms
2. Severity
3. Duration
4. Triggers
5. Associated symptoms

Known symptoms:
{state.symptoms_positive}

Conversation history:
{state.get_conversation_history()}

TASK:
Ask the most useful next question to improve diagnosis accuracy.

Return ONLY the question.
"""
        return call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()

    # First turn -> greeting
    if state.turn_count == 0:
        return "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"

    # For all subsequent turns, ensure input is health-related
    user_input = state.answers[-1] if state.answers else ""
    if user_input and not is_health_related(user_input):
        return get_domain_redirect_response()

    feedback_context = ""
    feedbacks = getattr(state, "verification_feedbacks", [])
    if feedbacks:
        latest = feedbacks[-1]
        feedback_context = f"Supervisor feedback from Agent 2:\n{latest}\n"

    prompt = f"""
You are Agent 1, a professional Ayurvedic medical interviewer.

GOAL:
Collect accurate clinical information needed for diagnosis.

REASONING INSTRUCTION:
Analyze the symptoms, OPQRST information, and conversation history carefully.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

RULES:
- Ask only ONE question at a time
- Avoid repeating questions
- Prefer short clear questions
- Focus on missing diagnostic information
- Use a mix of YES/NO and short open questions
STRICT CLINICAL RULES:
1. Never invent symptoms
2. If the patient denies a symptom, exclude it
3. Use only explicitly stated symptoms
4. Do not reinterpret answers

PRIORITY INFORMATION:
1. Location of symptoms
2. Severity
3. Duration
4. Triggers
5. Associated symptoms

Known symptoms:
{state.symptoms_positive}

STRUCTURED DATA (use exact values, do not change):
Severity: {state.collected_fields.get('severity','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Time: {state.collected_fields.get('time','unknown')}

Conversation history:
{state.get_conversation_history()}

{feedback_context}TASK:
Ask the most useful next question to improve diagnosis accuracy.

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
    data_completeness = min(len(state.symptoms_positive) / len(required_fields), 1.0) if required_fields else 0
    
    opqrst_context = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""
    
    if data_completeness < 0.6 or len(state.symptoms_positive) < 2:
        print("\nWARNING: Agent 1 needs more data to diagnose realistically. Continuing with more questions...\n")
    
    prompt = f"""
You are Agent 1 performing preliminary Ayurvedic diagnostic reasoning.

REASONING INSTRUCTION:
Carefully analyze the symptoms and OPQRST information before generating diagnoses.
CRITICAL RULE:
Use ONLY the structured data below.
Do NOT extract symptoms from conversation history.
Do NOT reinterpret answers.

Use ONLY the structured data below.
Do NOT extract new information from conversation history.

KNOWN SYMPTOMS:
{state.symptoms_positive}

OPQRST INFORMATION:
{opqrst_context}

Structured Data:
Onset: {state.collected_fields.get('onset','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Severity: {state.collected_fields.get('severity','unknown')}
Time: {state.collected_fields.get('time','unknown')}

DOSHA SCORES:
Vata: {dosha_scores.get('vata',0)}
Pitta: {dosha_scores.get('pitta',0)}
Kapha: {dosha_scores.get('kapha',0)}

DATA COMPLETENESS:
{int(data_completeness*100)}%

STRUCTURED DATA (use exact values, do not change):
Severity: {state.collected_fields.get('severity','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Time: {state.collected_fields.get('time','unknown')}

STRICT CLINICAL RULES:
1. Never invent symptoms
2. If the patient denies a symptom, exclude it
3. Use only explicitly stated symptoms
4. Do not reinterpret answers

STRUCTURED REASONING STEPS:
STEP 1: Summarize confirmed symptoms
STEP 2: Identify symptom patterns
STEP 3: Generate differential diagnoses
STEP 4: Map to Ayurvedic dosha imbalance
STEP 5: Produce confidence score

TASK:
Generate possible Ayurvedic conditions that best match the symptoms.

RULES:
- Only use conditions described in classical Ayurveda
- Diagnosis must logically follow the symptoms
- If data is incomplete, lower the confidence score
- Provide differential diagnoses
- First identify the biomedical symptom pattern, then map to Ayurvedic interpretation

OUTPUT FORMAT:

DIFFERENTIAL DIAGNOSES:
1. Condition - Probability% - reasoning
2. Condition - Probability% - reasoning
3. Condition - Probability% - reasoning

DOSHA IMBALANCE:
Explain the dominant dosha pattern

CONFIDENCE SCORE:
Number between 0.0 and 1.0
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


def ask_generated_yes_no_questions(state, questions):
    """
    Ask a list of YES/NO questions generated by Agent 2.
    This is used when confidence is low and Agent 2 generates clarification questions.
    Returns the number of questions answered.
    """
    
    print(f"\n--- [Agent 1: Asking {len(questions)} Clarification Questions to Increase Confidence] ---\n")
    
    for i, question in enumerate(questions, 1):
        print(f"Agent 1 (Q{i}/{len(questions)}): {question}")
        answer = input("User (YES/NO): ").strip()
        state.record_interaction(question, answer)
    
    print(f"\nCompleted {len(questions)} clarification questions.")
    return len(questions)
