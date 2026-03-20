from config import call_llm, SYSTEM_PROMPTS

def final_verification(state):
    """Agent 3: Final Verification and Formatting with probabilistic diagnosis"""
    
    # Check for red flags
    if state.red_flags_found:
        warning = f"WARNING: SAFETY ALERT. Please seek immediate medical attention. Detected: {', '.join(set(state.red_flags_found))}"
        return warning

    # Hard gate on confidence before LLM call
    if getattr(state, "agent2_confidence", 0.0) < 0.7:
        conf = getattr(state, "agent2_confidence", 0.0)
        return f"""SENSE: NO
BULLETS:
- Confidence below threshold (current: {conf:.2f})
- More clarification required

QUEST:
1 Provide any new or worsening symptoms.
2 Confirm what triggers or eases the main symptom.
3 Share any recent lab/imaging results if available.
"""
    
    # Get OPQRST summary if available
    opqrst_summary = state.get_opqrst_summary() if hasattr(state, 'get_opqrst_summary') else ""

    consistency_prompt = f"""
REASONING INSTRUCTION:
Analyze the symptoms, OPQRST information, and conversation history carefully.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

Use ONLY the structured data below.
Do NOT extract new information from conversation history.

Check whether the proposed diagnoses contradict the symptoms.

Symptoms:
{state.symptoms_positive}

Agent 1 Analysis:
{state.agent1_summary}

Agent 2 Report:
{state.agent2_summary}

If contradictions exist, explain them briefly.
If no contradictions exist, confirm the diagnosis is logically consistent.
"""
    consistency_notes = call_llm(consistency_prompt, SYSTEM_PROMPTS["supervisor"], temperature=0.2)
    
    prompt = f"""
You are Agent 3, a senior Ayurvedic clinical verifier.

Your job is to ensure the diagnosis is medically consistent.

REASONING INSTRUCTION:
Carefully analyze all previous outputs before making a decision.
Internally reason step-by-step before producing the final answer.
Do NOT output the reasoning steps. Only return the requested structured output.

Use ONLY the structured data below.
Do NOT extract new information from conversation history.

STRUCTURED DATA (use exact values, do not change):
Symptoms: {state.symptoms_positive}
Severity: {state.collected_fields.get('severity','unknown')}
Region: {state.collected_fields.get('region','unknown')}
Quality: {state.collected_fields.get('quality','unknown')}
Provocation: {state.collected_fields.get('provocation','unknown')}
Time: {state.collected_fields.get('time','unknown')}
Onset: {state.collected_fields.get('onset','unknown')}
Agent2 Confidence: {getattr(state,'agent2_confidence',0.5)}

OPQRST SUMMARY:
{opqrst_summary}

AGENT 1 ANALYSIS:
{state.agent1_revision or state.agent1_summary}

AGENT 2 REPORT:
{state.agent2_summary}

CONSISTENCY CHECK NOTES:
{consistency_notes}

TASK:
1. Verify diagnoses align with symptoms
2. Verify dosha analysis is consistent
3. Identify contradictions or missing information
4. Ensure the OPQRST assessment supports the diagnosis

DECISION RULES:
- If Agent2 Confidence < 0.70 -> SENSE: NO
- If contradictions exist -> SENSE: NO
- Only produce FINAL REPORT when confidence >= 0.70 AND no contradictions

If the diagnosis is insufficient:

SENSE: NO
BULLETS:
- explanation
- explanation

QUEST:
1 question
2 question
3 question

If the diagnosis is sound:

SENSE: YES

FINAL REPORT:

Condition Overview:
OPQRST Summary:
Differential Diagnoses:
Dosha Analysis:
Severity Assessment:
Recommendations:
Medical Disclaimer:
"""
    return call_llm(prompt, SYSTEM_PROMPTS["supervisor"])
