from config import call_llm, SYSTEM_PROMPTS


def verify_questions(state):
    """
    Agent 2 (Supervisor):
    Reviews Agent 1's questioning strategy and enforces improvements.
    """

    history = state.get_conversation_history()

    prompt = f"""
You are Agent 2, a Senior Ayurvedic Medical Supervisor.

CRITICAL: Your response must be MAX 200 WORDS.

Evaluate Agent 1's questioning strategy:

Conversation History:
{history}

DATA SUFFICIENCY RULE:
If Symptoms + all OPQRST fields are present, consider data sufficient and DO NOT ask for more questions.

Evaluate using this STRICT format:

STRENGTHS: (max 2 points)
- What was done well

MISSING: (max 3 items)
- Crucial information still needed

ACTION REQUIRED:
- If confidence < 0.6: "Ask more questions to gather sufficient data"
- If symptoms unclear: "Clarify specific symptom details"
- If red flags present: "URGENT: Check for safety concerns"

Return ONLY this format.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["validator"])


def create_final_summary(state):
    """
    Agent 2:
    Generates a comprehensive clinical report with differential diagnoses.
    Returns both the summary and the confidence score.
    """

    # Get OPQRST summary if available
    opqrst_summary = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""

    prompt = f"""
You are Agent 2, a Senior Ayurvedic Clinical Reviewer.

Your role is to critically evaluate Agent 1's diagnostic reasoning.

REASONING INSTRUCTION:
Carefully analyze the symptoms, OPQRST assessment, and Agent 1 conclusions.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

CRITICAL RULE:
Use ONLY the structured data below.
Do NOT extract new information from conversation history.
Do NOT reinterpret answers.
Do NOT generate confidence scores. Confidence is calculated by the system.
Do NOT claim something is missing if it exists in structured data.
DATA SUFFICIENCY RULE:
If the following are present, consider data sufficient and do NOT request more questions:
- Symptoms
- All OPQRST fields
- Diet, Digestion, Sleep, Energy level, Tongue description (if provided)

OPQRST ASSESSMENT:
{opqrst_summary}

SYMPTOMS:
{state.symptoms_positive}

AGENT 1 ANALYSIS:
{state.agent1_summary}

STRUCTURED DATA (use exact values, do not change):
Onset: {state.collected_fields.get('onset','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Severity: {state.collected_fields.get('severity','unknown')}
Time: {state.collected_fields.get('time','unknown')}
Additional Symptoms: {state.symptoms_positive}

STRICT CLINICAL RULES:
1. Never invent symptoms
2. If the patient denies a symptom, exclude it
3. Use only explicitly stated symptoms
4. Do not reinterpret answers
5. Use ONLY the structured data below; do NOT extract new info from conversation history.
 6. Do NOT output confidence.

STRUCTURED REASONING STEPS:
STEP 1: Summarize confirmed symptoms
STEP 2: Identify symptom patterns
STEP 3: Generate differential diagnoses
STEP 4: Map to Ayurvedic dosha imbalance
STEP 5: Produce confidence score

TASK:
1. Evaluate whether the diagnoses logically match the symptoms
2. Identify contradictions or missing information
3. Generate refined differential diagnoses

RULES:
- Use Ayurvedic clinical logic
- Ensure OPQRST data supports the diagnoses
- If information is insufficient, lower the confidence score

OUTPUT FORMAT:

POSSIBLE CONDITIONS:
1. Condition - X% confidence - reason
2. Condition - X% confidence - reason
3. Condition - X% confidence - reason

DOSHA ANALYSIS:
Dominant dosha:
Imbalance explanation:

CONFIDENCE SCORE:
(System will add)

MISSING INFORMATION:
- item
- item

RECOMMENDATIONS:
- diet
- lifestyle
"""

    response = call_llm(prompt, SYSTEM_PROMPTS["validator"])
    return response, 0.0  # Confidence calculated deterministically elsewhere


def _extract_confidence(response_text):
    """Deprecated: confidence now computed deterministically"""
    return 0.0


def generate_clarification_questions(state):
    """
    Agent 2:
    When confidence < 0.7, generate 5-6 YES/NO questions to increase confidence.
    Returns a list of questions.
    """

    history = state.get_conversation_history()
    opqrst_summary = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""

    prompt = f"""
You are a senior Ayurvedic diagnostic reviewer.

The current diagnostic confidence is LOW.

REASONING INSTRUCTION:
Analyze the symptoms, OPQRST summary, and conversation history carefully.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

Use ONLY the structured data below.
Do NOT extract new information from conversation history.

KNOWN SYMPTOMS:
{state.symptoms_positive}

OPQRST SUMMARY:
{opqrst_summary}

STRUCTURED DATA (use exact values, do not change):
Onset: {state.collected_fields.get('onset','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Severity: {state.collected_fields.get('severity','unknown')}
Time: {state.collected_fields.get('time','unknown')}
Additional Symptoms: {state.symptoms_positive}

TASK:
Generate exactly 6 YES/NO questions.

RULES:
- Each question must help confirm or eliminate a diagnosis
- Questions must be short
- Focus on missing or ambiguous symptoms
- Do not request information already answered in the conversation
- Apply DATA SUFFICIENCY RULE: If Symptoms AND all OPQRST fields are present, skip generating questions.

OUTPUT:

QUESTIONS:
1.
2.
3.
4.
5.
6.
"""

    response = call_llm(prompt, SYSTEM_PROMPTS["validator"], temperature=0.5)
    
    # Parse questions from response
    questions = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and len(line) > 0:
            # Check if line starts with a number
            parts = line.split('.', 1)
            if len(parts) == 2 and parts[0].isdigit():
                question = parts[1].strip()
                if question and len(question) > 3:
                    questions.append(question)
    
    return questions[:6]  # Return max 6 questions
