from config import call_llm, SYSTEM_PROMPTS

def final_verification(state):
    """Agent 3: Final Verification and Formatting."""
    prompt = f"""
    You are Agent 3 (The Final Verifier).
    
    Review:
    1. Chat History Summary 1 (Agent 1): {state.agent1_summary}
    2. Summary 2 (Agent 2): {state.agent2_summary}
    3. Conversation: {state.get_conversation_history()}
    
    Check if the diagnosis makes sense.Understand the chat history.
    Check if the symptoms and the diagnosis aligns well with respect to AYURVEDIC TEXTS 100%. If it doesn't make sense, provide specific bullet points on why it doesn't make sense and what kind of questions can be asked by Agent 1 to fix this.
    If it does make sense, format the final output beautifully for the user, including information about the disease, how dangerous it is, and ensure the presentation is professional and well-structured without any internal agent chatter.  
    Try to use AYURVEDIC TEXTS as much as possible as the basis for your reasoning and final output.
    Ensure u understand the Agent 1 
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
    - No MARKDOWN, just plain text.
    - USE AYURVEDIC TEXTS as the basis for your reasoning and final output.
    """
    return call_llm(prompt, SYSTEM_PROMPTS["supervisor"])
