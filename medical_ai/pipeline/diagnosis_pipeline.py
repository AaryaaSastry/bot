from medical_ai.memory.diagnostic_state import DiagnosticState
from medical_ai.agents.interviewer import ask_question, generate_simple_diagnosis, ask_opqrst_question
from medical_ai.agents.validator import verify_questions, create_final_summary
from medical_ai.agents.supervisor import final_verification
from medical_ai.medical_framework import DiagnosisRefusal
from config import call_llm, SYSTEM_PROMPTS, MAX_QUESTIONS, SUPERVISOR_INTERVAL

def _extract_opqrst_from_text(state, text):
    """Extract OPQRST fields from user text using LLM"""
    
    prompt = f"""
You are an OPQRST medical analyzer. Extract the following fields from the patient's response:
- onset: When did symptoms start? (e.g., "3 days ago", "yesterday", "last week")
- provocation: What makes it worse or better?
- quality: Description of symptom (e.g., "vomiting", "sharp pain", "dull ache")
- region: Location of symptom (e.g., "stomach", "head", "chest")
- severity: How bad is it? (1-10 scale)
- time: How long does it last? / How often does it occur?

Patient response: "{text}"

Return ONLY the fields you can identify in this format:
field: value

If a field cannot be determined, skip it.
"""
    
    response = call_llm(prompt, "You are a medical data extractor.", temperature=0.3)
    
    # Parse the response and record fields
    for line in response.strip().split('\n'):
        if ':' in line:
            field, value = line.split(':', 1)
            field = field.strip().lower()
            value = value.strip()
            if field in DiagnosticState.OPQRST_FIELDS and value:
                state.record_opqrst_field(field, value)

def run_session():
    state = DiagnosticState()
    print("\n--- [AI Medical Diagnostic System Initialized] ---\n")
    
    # ============================================================================
    # PHASE 1: COLLECT OPQRST FIELDS (Mandatory)
    # ============================================================================
    print("\n--- [Phase 1: OPQRST Assessment - Collecting Required Information] ---\n")
    
    # Ask initial greeting
    initial_greeting = "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
    print(f"Agent 1: {initial_greeting}")
    answer = input("User: ")
    state.record_interaction(initial_greeting, answer)
    
    # Extract OPQRST fields from the initial response
    print("\n[Analyzing your response for OPQRST details...]")
    _extract_opqrst_from_text(state, answer)
    
    # Show what we've extracted
    if state.collected_fields:
        print(f"   [Already collected: {', '.join([f for f, v in state.collected_fields.items() if v])}]")
    
    # Collect remaining missing OPQRST fields
    missing = state.get_missing_opqrst_fields()
    if missing:
        print(f"   [Still need: {', '.join(missing)}]\n")
    while not state.is_opqrst_complete():
        opqrst_result = ask_opqrst_question(state)
        
        if opqrst_result is None:
            break  # All fields collected
        
        question = opqrst_result["question"]
        field = opqrst_result["field"]
        
        print(f"Agent 1 (OPQRST): {question}")
        answer = input("User: ")
        state.record_interaction(question, answer)
        
        # Record the OPQRST field
        state.record_opqrst_field(field, answer)
        
        print(f"   [OPQRST Progress: {len(state.OPQRST_FIELDS) - len(state.get_missing_opqrst_fields())}/{len(state.OPQRST_FIELDS)} fields collected]")
    
    print(f"\n✅ OPQRST Assessment Complete! All {len(state.OPQRST_FIELDS)} fields collected.")
    
    # ============================================================================
    # PHASE 2: MAIN INTERVIEW (Agent 1 asks questions)
    # ============================================================================
    print("\n--- [Phase 2: Detailed Interview - Agent 1] ---\n")
    
    # Agent 1 asks MAX_QUESTIONS questions
    for i in range(MAX_QUESTIONS):
        question = ask_question(state)
        print(f"Agent 1: {question}")
        answer = input("User: ")
        state.record_interaction(question, answer)
        
        # Check for red flags after each response
        if state.red_flags_found:
            print(f"\n⚠️ SAFETY WARNING: Detected potential red flags: {', '.join(set(state.red_flags_found))}")
            print("Please consult a healthcare professional if symptoms are severe.")
        
        # Agent 2 verification every SUPERVISOR_INTERVAL questions (3 times for 15 questions)
        if (i + 1) % SUPERVISOR_INTERVAL == 0:
            print(f"\n--- [Agent 2 Verifying Questions (Cycle {state.cycle_count + 1})...] ---")
            verification = verify_questions(state)
            print(f"Agent 2: {verification}")
            
            if "APPRECIATION" not in verification:
                state.verification_feedbacks.append(verification)
            state.cycle_count += 1

    # Agent 1 generates a simple diagnosis
    print("\n--- [Agent 1 Generating Preliminary Diagnosis...] ---")
    state.agent1_summary = generate_simple_diagnosis(state)
    print(f"Agent 1 Summary: {state.agent1_summary}")

    # Agent 2 creates a summary for Agent 3
    print("\n--- [Agent 2 Creating Final Summary...] ---")
    state.agent2_summary = create_final_summary(state)

    # Agent 3: Final Verification
    print("\n--- [Agent 3 Final Verification...] ---")
    final_output = final_verification(state)

    if "SENSE: NO" in final_output:
        print("\n[Agent 3: Diagnosis needs correction!]")
        print(final_output)
        
        # 1. specific bullet points why it doesn't make sense 
        # 2. kind of questions by Agent 1 
        # 3. 5 extra YES/NO questions.
        state.extra_questions_asked = True
        
        for _ in range(5):
            question = ask_question(state)
            print(f"Agent 1 (Extra): {question}")
            answer = input("User: ")
            state.record_interaction(question, answer)
        
        # Final summary by Agent 2 (no more giving back to Agent 1)
        print("\n--- [Agent 2 Final Summary After Extra Questions...] ---")
        state.agent2_summary = create_final_summary(state)
        
        # Finally to Agent 3 for final presentation
        print("\n--- [Agent 3 Final Report Generation...] ---")
        final_output = final_verification(state)

    print("\n--- [FINAL CONSULTATION REPORT] ---\n")
    print(final_output)

if __name__ == "__main__":
    run_session()
