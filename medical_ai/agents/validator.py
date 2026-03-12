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
    """

    history = state.get_conversation_history()
    
    # Get OPQRST summary if available
    opqrst_summary = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""

    prompt = f"""
You are Agent 2, a Senior Ayurvedic Clinical Reviewer.

CRITICAL: 
- Use DIFFERENTIAL DIAGNOSIS (multiple possibilities with percentages)
- Keep response under 300 words

OPQRST ASSESSMENT (Required Fields):
{opqrst_summary}

Symptoms Identified:
{state.symptoms_positive}

Agent 1 Reasoning:
{state.agent1_summary}

Conversation History:
{history}

DISEASE VALIDITY RULE:
BASE RULE: ONLY CONSIDER THE SYMPTOMS THAT THE USER GAVE.
- Only mention Ayurvedic diseases from classical literature
- If no confident match, describe Dosha imbalance pattern
- ENSURE OPQRST fields are properly considered in the analysis
- MAKE SURE THAT THE DIAGNOSIS LOGICALLY FOLLOWS FROM THE SYMPTOMS AND OPQRST DATA. THEY SHOULD LOGICALLY ALIGN.
Create the final report using this STRUCTURED format:

POSSIBLE CONDITIONS:
1. [Condition] - [X]% confidence - [brief reason]
2. [Condition] - [X]% confidence - [brief reason]
3. [Condition] - [X]% confidence - [brief reason]

OPQRST ASSESSMENT SUMMARY:
- Onset: [value]
- Provocation: [value]
- Quality: [value]
- Region: [value]
- Severity: [value]
- Time: [value]

DOSHA ANALYSIS:
- Dominant dosha: [Vata/Pitta/Kapha]
- Imbalance pattern: [description]

RECOMMENDATIONS:
- [Diet]
- [Lifestyle]

SAFETY NOTE: [If urgent, state clearly]

Return ONLY this structure.
"""

    return call_llm(prompt, SYSTEM_PROMPTS["validator"])