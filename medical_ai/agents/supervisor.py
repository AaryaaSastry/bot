from config import call_gemini

def final_verification(state):
    """Agent 3: Final Verification and Formatting."""
    prompt = f"""
    You are Agent 3 (The Final Verifier).
    
    Review:
    1. Chat History Summary 1 (Agent 1): {state.agent1_summary}
    2. Summary 2 (Agent 2): {state.agent2_summary}
    3. Conversation: {state.get_conversation_history()}
    
    Check if the diagnosis makes senseUnderstand the chat history .
    
    IF DIAGNOSIS DOES NOT MAKE SENSE:
    Output exactly:
    'SENSE: NO'
    'BULLETS: [Specific bullet points why it doesn't make sense]'
    'QUEST: [What kind of questions can be asked by Agent 1 to fix this]'
    
    IF DIAGNOSIS MAKES SENSE:
    Output exactly:
    'SENSE: YES'
    'FINAL: [Format the final output beautifully for the user]'
    
    For the final user-facing output:
    - Include information about the disease.
    - How dangerous it is.
    - Presentation: Professional and well-structured.
    - No internal agent chatter.
    """
    return call_gemini(prompt)
