#static code- will need this for agent 4 integration 

from config import call_llm, SYSTEM_PROMPTS


def generate_diagnosis(state):
    prompt = f"""
    Complete conversation history:
    {state.get_chat_history_text()}

    Symptoms reported:
    {state.symptoms_positive}

    Symptoms denied:
    {state.symptoms_negative}

    REASONING INSTRUCTION:
    Analyze the symptoms, OPQRST information, and conversation history carefully.
    Internally reason step-by-step before producing the final answer.
    Do NOT output the reasoning steps. Only return the requested structured output.

    Use ONLY the structured data below.
    Do NOT extract new information from conversation history.

    STRUCTURED DATA (use exact values, do not change):
    Severity: {state.collected_fields.get('severity','unknown')}
    Region: {state.collected_fields.get('region','unknown')}
    Quality: {state.collected_fields.get('quality','unknown')}
    Provocation: {state.collected_fields.get('provocation','unknown')}
    Time: {state.collected_fields.get('time','unknown')}
    Onset: {state.collected_fields.get('onset','unknown')}
    Additional Symptoms: {state.symptoms_positive}

    STRICT CLINICAL RULES:
    1. Never invent symptoms
    2. If the patient denies a symptom, exclude it
    3. Use only explicitly stated symptoms
    4. Do not reinterpret answers
    5. Use ONLY the structured data below; do NOT extract new info from conversation history.

    STRUCTURED REASONING STEPS:
    STEP 1: Summarize confirmed symptoms
    STEP 2: Identify symptom patterns
    STEP 3: Generate differential diagnoses
    STEP 4: Map to Ayurvedic dosha imbalance
    STEP 5: Produce confidence score

    Based on these symptoms and the complete conversation, generate 3 possible diagnoses.

    For each diagnosis:
    - Name
    - Short explanation
    """

    diagnosis = call_llm(prompt, SYSTEM_PROMPTS["supervisor"])

    return diagnosis
