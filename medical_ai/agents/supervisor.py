from config import call_llm, SYSTEM_PROMPTS

def final_verification(state):
    """Agent 3: Final Verification and Formatting."""
    prompt = f"""
    You are Agent 3 (Final Ayurvedic Clinical Verifier).
    
    Review:
    1. Agent 1 Analysis: {state.agent1_summary}
    2. Agent 2 Clinical Report: {state.agent2_summary}
    3. Conversation History: {state.get_conversation_history()}
    
    Validation Tasks:
    - Verify if the primary diagnosis aligns with classical Ayurvedic texts (Charaka, Sushruta, etc.).
    - Ensure symptoms, Agni status, and Dosha dynamics are consistent.
    - If the diagnosis is contradictory or lacks evidence, provide specific bullet points for clarification.
    - If valid, format a professional, user-friendly final output.

    IF DIAGNOSIS DOES NOT MAKE CLINICAL SENSE OR LACKS VITAL INFO (SENSE: NO):
    - Output ONLY:
    'SENSE: NO'
    'BULLETS: [Precise points on why it doesn't align with Ayurvedic principles]'
    'QUEST: [3 refined clinical questions for Agent 1 to ask]'
    
    IF DIAGNOSIS IS CLINICALLY SOUND (SENSE: YES):
    - Output ONLY:
    'SENSE: YES'
    'FINAL: [A high-end, professional clinical summary for the user]'

    Final Output Format (for SENSE: YES):
    - DISEASE OVERVIEW: Name and brief Ayurvedic description (Pathogenesis).
    - DOSHA-DUSHYA DYNAMICS: Briefly explain the imbalance in simple terms.
    - SEVERITY & GUIDELINES: Explain current severity and general Ayurvedic advice (lifestyle/food).
    - DISCLAIMER: Standard medical disclaimer that this is not a substitute for professional clinical advice.

    General Requirements:
    - PROFESSIONAL tone.
    - NO internal agent chatter.
    - NO Markdown formatting, plain text only.
    - GROUNDED in Ayurvedic tradition.
    """
    return call_llm(prompt, SYSTEM_PROMPTS["supervisor"])
