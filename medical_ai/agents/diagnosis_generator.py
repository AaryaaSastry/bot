from config import call_gemini


def generate_diagnosis(state):
    prompt = f"""
    Complete conversation history:
    {state.get_chat_history_text()}

    Symptoms reported:
    {state.symptoms_positive}

    Symptoms denied:
    {state.symptoms_negative}

    Based on these symptoms and the complete conversation, generate 3 possible diagnoses.

    For each diagnosis:
    - Name
    - Short explanation
    """

    diagnosis = call_gemini(prompt)

    return diagnosis