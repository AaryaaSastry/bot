from config import call_llm, SYSTEM_PROMPTS

def final_verification(state):
    """Agent 3: Final Verification and Formatting with probabilistic diagnosis"""
    
    # Check for red flags
    if state.red_flags_found:
        warning = f"⚠️ SAFETY ALERT: Please seek immediate medical attention. Detected: {', '.join(set(state.red_flags_found))}"
        return warning
    
    # Get OPQRST summary if available
    opqrst_summary = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""
    
    prompt = f"""
You are Agent 3 (Final Ayurvedic Clinical Verifier).

CRITICAL: 
- Keep output under 800 words
- Use differential diagnoses with percentages
- If confidence < 0.6, state "More information needed"

OPQRST ASSESSMENT (Mandatory Fields):
{opqrst_summary}

Review:
1. Agent 1 Analysis: {state.agent1_summary}
2. Agent 2 Clinical Report: {state.agent2_summary}
3. Conversation History: {state.get_conversation_history()}

VALIDATION TASKS:
1. Check if diagnosis aligns with classical Ayurveda
2. Verify Dosha consistency
3. Identify missing info
4. Check for contradictions
5. ENSURE OPQRST fields are properly considered in diagnosis

DECISION RULE:

If diagnosis insufficient:
SENSE: NO
BULLETS: [reasons]
QUEST: [3 questions to clarify]

If diagnosis sound:
SENSE: YES
FINAL: [Professional report with:
- Condition Overview
- OPQRST Assessment Summary (must include all 6 fields)
- Differential Diagnoses (with %)
- Dosha Analysis
- Severity
- Recommendations
- Disclaimer]

Output exactly in this format.
"""
    return call_llm(prompt, SYSTEM_PROMPTS["supervisor"])
