from config import call_llm, SYSTEM_PROMPTS


def is_health_related(user_input):
    """Check if user input is health-related using simple keyword matching"""
    
    if not user_input:
        return False
    
    user_input_lower = user_input.lower()
    
    # Keywords that indicate health-related input
    health_keywords = [
        # Symptoms and sensations
        "pain", "hurt", "ache", "feeling", "felt", "feel", "nausea", "dizzy", "dizziness",
        "tired", "fatigue", "exhausted", "weak", "weakness", "fever", "chills", "cold", "hot",
        "headache", "stomach", "chest", "back", "joint", "muscle", "bone", "skin", "rash",
        "itch", "tingling", "numb", "burning", "throb", "pulse", "heart", "beat", "breath",
        "cough", "sneeze", "sore", "throat", "nose", "ear", "eye", "tooth", "toothache",
        "vomit", "vomiting", "diarrhea", "constipation", "stool", "urine", "bleed", "bleeding",
        "swell", "swelling", "lump", "bump", "growth", "spot", "mark", "discharge", "odor",
        "smell", "taste", "appetite", "hunger", "thirst", "sleep", "insomnia", "dream",
        "weight", "gain", "loss", "increase", "decrease", "change", "problem", "issue",
        "symptom", "condition", "disease", "illness", "sick", "ill", "healthy", "health",
        "medical", "medicine", "doctor", "hospital", "clinic", "treatment", "therapy",
        "prescription", "tablet", "capsule", "syrup", "injection", "dose", "medication",
        # Ayurvedic terms
        "dosha", "pitta", "kapha", "vata", "ayurved", "ayurvedic", "prakriti",
        "digestion", "digest", "indigestion", "gas", "bloating", "acidity", "acid",
        # Body parts
        "head", "neck", "shoulder", "arm", "hand", "finger", "leg", "knee", "foot", "toe",
        "face", "mouth", "tongue", "lip", "jaw", "cheek", "forehead", "brain", "nerve",
        "lung", "liver", "kidney", "intestine", "bowel", "colon", "bladder",
        "blood", "spine", "spinal", "cardiac", "vascular",
        # General health questions
        "cause", "reason", "why", "what is", "how to", "cure", "remedy", "relief",
        "prevent", "avoid", "reduce", "improve", "worsen", "better", "worse", 
        "normal", "abnormal", "serious", "mild", "severe", "chronic", "acute"
    ]
    
    # Check if any health keyword is present
    for keyword in health_keywords:
        if keyword in user_input_lower:
            return True
    
    return False


def evaluate_symptom_severity(user_input, history):
    """Evaluate symptom severity using keyword matching instead of LLM"""
    
    if not user_input:
        return "LOW", "No symptoms detected"
    
    user_input_lower = user_input.lower()
    
    # High severity keywords - potential emergencies
    high_severity_keywords = [
        "severe", "extreme", "unbearable", "emergency", "urgent", "critical",
        "chest pain", "heart attack", "stroke", "blood in stool", "vomiting blood",
        "black stool", "fainting", "unconscious", "difficulty breathing", "cannot breathe",
        "suicide", "overdose", "poison", "severe bleeding", "internal bleeding",
        "sudden onset", "worst pain", "life threatening"
    ]
    
    # Medium severity keywords
    medium_severity_keywords = [
        "moderate", "significant", "worsening", "persistent", "recurring",
        "high fever", "dehydration", "dizzy", "weakness"
    ]
    
    # Check high severity first
    for keyword in high_severity_keywords:
        if keyword in user_input_lower:
            return "HIGH", f"Detected high severity keyword: {keyword}"
    
    # Check medium severity
    for keyword in medium_severity_keywords:
        if keyword in user_input_lower:
            return "MODERATE", f"Detected medium severity keyword: {keyword}"
    
    return "LOW", "No high-risk keywords detected"


def handle_domain_logic(state, user_input):
    """Handle domain awareness and redirection logic"""
    health_related = is_health_related(user_input)

    # CASE 1 — conversation has not entered medical domain yet
    if not getattr(state, "medical_topic_confirmed", False):

        if not health_related:
            return (
                "I'm designed to assist with health concerns. "
                "Could you please describe any symptoms or medical issues you're experiencing?"
            )
        else:
            state.medical_topic_confirmed = True
            return None

    # CASE 2 — mid conversation off-topic
    if getattr(state, 'medical_topic_confirmed', False) and not health_related:

        if getattr(state, 'last_valid_question', None):
            return (
                "I believe this may be outside my capabilities to assist with. "
                "Let's focus on your health concern.\n\n"
                f"{state.last_valid_question}"
            )

        return (
            "Let's focus on your health concern. "
            "Could you describe your symptoms again?"
        )

    return None


def get_domain_redirect_response():
    """Return a polite response asking user to discuss only health problems"""
    return "I appreciate your question, but I'm specifically designed to help with health-related concerns and medical symptoms. Please tell me about any health problems or symptoms you're experiencing, and I'll be happy to assist you from an Ayurvedic perspective."


def detect_red_flags(user_input):
    """Detect dangerous symptoms that require urgent medical care"""

    red_flags = [
        "blood in stool",
        "vomiting blood",
        "black stool",
        "severe abdominal pain",
        "fainting",
        "unconscious",
        "severe chest pain",
        "difficulty breathing",
    ]

    text = user_input.lower()

    for flag in red_flags:
        if flag in text:
            return True

    return False


def _red_flag_or_redirect(state, user_input):
    """Return an immediate response if dangerous severity or non-health input is detected."""
    if not user_input:
        return None
        
    # Check severity dynamically instead of hardcoded red flags
    severity, explanation = evaluate_symptom_severity(user_input, state.get_conversation_history())
    if severity == "HIGH":
        return "Your symptoms may indicate a serious condition. Please seek immediate medical attention or visit the nearest hospital."
    
    # Check domain logic
    redirect = handle_domain_logic(state, user_input)
    if redirect:
        return redirect
        
    return None


def ask_question(state):
    """Agent 1: Professional Medical Interviewer"""

    user_input = state.answers[-1] if state.answers else ""
    
    # Initial greeting handled separately if turn_count is 0
    if state.turn_count == 0:
        greeting = "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
        return greeting

    # Domain and safety checks for all other turns
    immediate = _red_flag_or_redirect(state, user_input)
    if immediate:
        return immediate

    # Phase 1: If diagnosis was rejected, ask clarification questions
    if getattr(state, "extra_questions_asked", False):
        prompt = f"""
You are Agent 1, a skilled, mindful, ayurvedic medical interviewer.

A previous diagnosis was rejected and you must clarify symptoms.

Conversation History:
{state.get_conversation_history()}

INTERVIEW GUIDELINES:
1. Focus strictly on symptoms described by the patient.
2. Ask only medically relevant questions.
3. If the user provides unrelated information, gently redirect them back to their symptoms.
4. Maintain context from previous questions.
5. Each question must help narrow the diagnosis.

QUESTION STYLE RULES:
- Ask ONE clear medical question at a time.
- Prefer short and specific answers.
- Yes/no questions are allowed when appropriate.
- However, allow descriptive responses when necessary (for example when asking about digestion, appetite, stool characteristics, or sleep patterns).
- Do NOT ask multiple questions at once.
- Avoid repeating previous questions.
- Focus on identifying Ayurvedic diagnostic signals such as:
  - appetite strength
  - digestion after meals
  - bloating or heaviness
  - stool consistency and frequency
  - thirst levels
  - sleep quality
  - body temperature tendencies
  - food cravings
Ensure the question is relevant to the symptoms and has not been asked before.
If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"

Return ONLY the question.
"""
        question = call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()
        state.last_valid_question = question
        state.question_count += 1
        return question

    # Second turn -> determine if user provided symptoms
    if state.turn_count == 1:
        prompt = f"""
You are Agent 1 (Medical Interviewer).

User response:
{state.answers[-1] if state.answers else ""}

Conversation History:
{state.get_conversation_history()}

INTERVIEW GUIDELINES:
1. Focus strictly on symptoms described by the patient.
2. Ask only medically relevant questions.
3. If the user provides unrelated information, gently redirect them back to their symptoms.
4. Maintain context from previous questions.
5. Each question must help narrow the diagnosis.

BASE RULE:
    - You are an ayurvedic medical interviewer, not a general chatbot. Always steer the conversation towards understanding the user's medical symptoms and concerns.
    - You need to understand the user properly like an experienced ayurvedic doctor, and not just list the symptoms. You need to take decent analysis of the symptoms and their patterns, and not just list them.
    - If the user asks about specific medications like paracetamol or any non-ayurvedic treatments, inform them that you can't advise on that as it is outside your ayurvedic expertise. Redirect them to describe their symptoms so you can help them from an ayurvedic perspective.

If the user only greeted you (like "hi", "hello"), politely ask them to describe their medical symptoms.
Greet the user if they greeted you, but also ask them to describe their symptoms. For example, you could say "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
If the user mentions their name, greet them by name and then ask about their symptoms that time only. For example, "Hello [Name]! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
If they mentioned symptoms, ask ONE relevant follow-up question to clarify them.

QUESTION STYLE RULES:
- Ask ONE clear medical question at a time.
- Prefer short and specific answers.
- Yes/no questions are allowed when appropriate.
- However, allow descriptive responses when necessary (for example when asking about digestion, appetite, stool characteristics, or sleep patterns).
- Do NOT ask multiple questions at once.
- Avoid repeating previous questions.
- Focus on identifying Ayurvedic diagnostic signals such as:
  - appetite strength
  - digestion after meals
  - bloating or heaviness
  - stool consistency and frequency
  - thirst levels
  - sleep quality
  - body temperature tendencies
  - food cravings
Ensure the question is relevant to the symptoms and has not been asked before.
If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"
Return ONLY the question.
"""
        question = call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()
        state.last_valid_question = question
        state.question_count += 1
        return question

    # Later turns -> follow-up questions
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

INTERVIEW GUIDELINES:
1. Focus strictly on symptoms described by the patient.
2. Ask only medically relevant questions.
3. If the user provides unrelated information, gently redirect them back to their symptoms.
4. Maintain context from previous questions.
5. Each question must help narrow the diagnosis.

QUESTION STYLE RULES:
- Ask ONE clear medical question at a time.
- Prefer short and specific answers.
- Yes/no questions are allowed when appropriate.
- However, allow descriptive responses when necessary (for example when asking about digestion, appetite, stool characteristics, or sleep patterns).
- Do NOT ask multiple questions at once.
- Avoid repeating previous questions.
- Focus on identifying Ayurvedic diagnostic signals such as:
  - appetite strength
  - digestion after meals
  - bloating or heaviness
  - stool consistency and frequency
  - thirst levels
  - sleep quality
  - body temperature tendencies
  - food cravings
- Ensure the question is clear and focused on one symptom or risk factor at a time.
- If there any big terminology, explain it in simple terms within the question, or use an example to clarify. For example, if asking about "dyspnea", you could say "Do you experience shortness of breath, especially when lying down or exerting yourself?"
Return ONLY the question text.
"""

    question = call_llm(prompt, SYSTEM_PROMPTS["interviewer"]).strip()
    state.last_valid_question = question
    state.question_count += 1
    return question


def generate_simple_diagnosis(state):
    """Agent 1: Preliminary diagnostic reasoning"""

    prompt = f"""
You are Agent 1, performing an Ayurvedic diagnostic analysis.

Your reasoning must follow classical Ayurvedic principles.

Known symptoms:
{state.symptoms_positive}

Conversation History:
{state.get_conversation_history()}

AYURVEDIC DIAGNOSTIC RULES:

Only use recognized Ayurvedic diagnostic categories such as:

Agni Disorders:
- Mandagni (weak digestion)
- Tikshnagni (excess digestive fire)
- Vishamagni (irregular digestion)
- Samagni (balanced digestion)

Digestive Disorders:
- Amlapitta
- Grahani
- Ajirna
- Ama accumulation

Dosha Patterns:
- Vata imbalance
- Pitta aggravation
- Kapha accumulation
- Mixed dosha imbalance

Do NOT invent disease names.

Your reasoning must explain:
- Which dosha is aggravated
- Whether Agni is weak, strong, or irregular
- Whether Ama accumulation is present
- How the symptoms match classical Ayurvedic patterns

Output format:

POSSIBLE CONDITION:
Name of Ayurvedic condition.

REASONING:
Explain the dosha imbalance and digestive pattern using Ayurvedic logic only.

Return ONLY this analysis.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["interviewer"])
