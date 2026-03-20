import json
from medical_ai.memory.diagnostic_state import DiagnosticState
from medical_ai.agents.interviewer import ask_question, generate_simple_diagnosis, ask_opqrst_question, ask_generated_yes_no_questions, revise_diagnosis
from medical_ai.agents.validator import verify_questions, create_final_summary, generate_clarification_questions
from medical_ai.agents.supervisor import final_verification
from medical_ai.medical_framework import DiagnosisRefusal, ConfidenceCalculator, detect_contradictions, RedFlagDetector
from config import call_llm, SYSTEM_PROMPTS, MAX_QUESTIONS, SUPERVISOR_INTERVAL
CLARIFICATION_MAX_CYCLES = 3
SUFFICIENCY_FIELDS = ["onset", "provocation", "quality", "region", "severity", "time"]


def validate_opqrst(collected_fields):
    if collected_fields.get("time", "").strip().lower() == "never":
        collected_fields["time"] = "first occurrence / unclear duration"
    return collected_fields


def data_sufficient(state):
    """
    Enhanced sufficiency logic.
    - OPQRST must be complete (standard triage clinical base).
    - Symptoms must be discrete and present (>=2).
    - Checks for red flags which may trigger immediate triage refusal.
    """
    opqrst_complete = state.is_opqrst_complete()
    symptoms_ok = len(state.symptoms_positive) >= 2
    
    # Check for acute severity in the absence of other fields as a fallback
    severity_val = 0
    sev = state.collected_fields.get("severity", "0")
    try:
        severity_val = int(sev)
    except: pass
        
    return (opqrst_complete and symptoms_ok) or (severity_val >= 8 and opqrst_complete)


def _extract_opqrst_from_text(state, text):
    """Extract OPQRST fields from user text using structured JSON output"""
    
    prompt = f"""
You are an OPQRST medical analyzer. Extract fields from the patient's response into JSON.

JSON SCHEMA:
{{
  "onset": "When did symptoms start? (e.g., '3 days ago')",
  "provocation": "What makes it worse or better?",
  "quality": "Description (e.g., 'sharp pain')",
  "region": "Location (e.g., 'stomach')",
  "severity": "numeric 1-10 string",
  "time": "Duration (e.g., 'constant', 'comes and goes')"
}}

Patient response: "{text}"

Return ONLY valid JSON. Skip fields you cannot identify.
"""
    
    response = call_llm(prompt, "You are a medical data extractor returning JSON.", temperature=0.1)
    
    try:
        # Clean potential markdown code blocks
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()
            
        data = json.loads(json_str)
        for field, value in data.items():
            field = field.lower()
            if field in DiagnosticState.OPQRST_FIELDS and value:
                # Basic normalization for severity
                if field == "severity":
                    try:
                        # Extract first number found
                        import re
                        nums = re.findall(r'\d+', str(value))
                        if nums:
                            val_int = int(nums[0])
                            if 1 <= val_int <= 10:
                                value = str(val_int)
                    except: pass
                state.record_opqrst_field(field, str(value))
    except Exception as e:
        print(f"   [Error parsing structured OPQRST data: {e}]")


def _extract_ayurvedic_evidence(state, text):
    """Extract Ayurvedic-specific clinical evidence using structured JSON"""
    
    prompt = f"""
Extract Ayurvedic clinical markers from the response into JSON:
{{
  "tongue": "Color, coating, or appearance",
  "digestion": "Appetite, bowel habits, bloating",
  "sleep": "Quality, duration, or timing",
  "energy": "Day energy level"
}}

Patient response: "{text}"

Return ONLY valid JSON.
"""
    response = call_llm(prompt, "You are an Ayurvedic data extractor returning JSON.", temperature=0.1)
    try:
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3].strip()
        elif json_str.startswith("```"):
            json_str = json_str[3:-3].strip()

        data = json.loads(json_str)
        if 'tongue' in data: state.tongue_description = data['tongue']
        if 'digestion' in data: state.digestion_status = data['digestion']
        if 'sleep' in data: state.sleep_pattern = data['sleep']
        if 'energy' in data: state.energy_level = data['energy']
    except Exception as e:
        pass


def run_session():
    state = DiagnosticState()
    print("\n--- [AI Medical Diagnostic System Initialized] ---\n")
    
    try:
        # ============================================================================
        # PHASE 1: COLLECT OPQRST FIELDS (Mandatory)
        # ============================================================================
        print("\n--- [Phase 1: OPQRST Assessment - Collecting Required Information] ---\n")
        
        # Ask initial greeting
        initial_greeting = "Hello! I'm here to help you with your medical concerns. Could you please describe the symptoms you're experiencing?"
        print(f"Agent 1: {initial_greeting}")
        answer = input("User: ")
        state.record_interaction(initial_greeting, answer)
        
        # TRIAGE ESCALATION: Check for red flags immediately after first answer
        if state.red_flags_found:
            raise DiagnosisRefusal(RedFlagDetector.get_red_flag_warning(state.red_flags_found), "RED_FLAG")

        # Extract OPQRST fields from the initial response
        print("\n[Analyzing your response for OPQRST details...]")
        _extract_opqrst_from_text(state, answer)
        state.collected_fields = validate_opqrst(state.collected_fields)
        
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
            
            # TRIAGE ESCALATION: Check red flags during OPQRST questions
            if state.red_flags_found:
                raise DiagnosisRefusal(RedFlagDetector.get_red_flag_warning(state.red_flags_found), "RED_FLAG")

            # Record the OPQRST field
            state.record_opqrst_field(field, answer)
            state.collected_fields = validate_opqrst(state.collected_fields)
            
            print(f"   [OPQRST Progress: {len(state.OPQRST_FIELDS) - len(state.get_missing_opqrst_fields())}/{len(state.OPQRST_FIELDS)} fields collected]")
        
        print(f"\n✅ OPQRST Assessment Complete! All {len(state.OPQRST_FIELDS)} fields collected.")
        
        # ============================================================================
        # PHASE 2: MAIN INTERVIEW (Agent 1) + INITIAL DIAGNOSIS
        # ============================================================================
        print("\n--- [Phase 2: Detailed Interview - Agent 1] ---\n")
        
        for i in range(MAX_QUESTIONS):
            question = ask_question(state)
            print(f"Agent 1: {question}")
            answer = input("User: ")
            state.record_interaction(question, answer)
            
            # Check for Ayurvedic evidence in Agent 1 answers
            _extract_ayurvedic_evidence(state, answer)

            if state.red_flags_found:
                raise DiagnosisRefusal(RedFlagDetector.get_red_flag_warning(state.red_flags_found), "RED_FLAG")
            
            if (i + 1) % SUPERVISOR_INTERVAL == 0:
                print(f"\n--- [Agent 2 Verifying Questions (Cycle {state.cycle_count + 1})...] ---")
                verification = verify_questions(state)
                print(f"Agent 2: {verification}")
                if "APPRECIATION" not in verification:
                    state.verification_feedbacks.append(verification)
                state.cycle_count += 1

        # Agent 1 initial diagnosis
        print("\n--- [Agent 1 Generating Preliminary Diagnosis...] ---")
        state.agent1_summary = generate_simple_diagnosis(state)
        print(f"Agent 1 Summary: {state.agent1_summary}")

        # Agent 2 critique
        print("\n--- [Agent 2 Creating Critique/Final Summary...] ---")
        state.agent2_summary, _ = create_final_summary(state)
        state.agent2_confidence = ConfidenceCalculator.stable_confidence(state)
        state.agent2_summary += f"\nCONFIDENCE SCORE: {state.agent2_confidence:.2f} (deterministic)"
        print(f"Agent 2 Summary (Confidence: {state.agent2_confidence:.1%}):\n{state.agent2_summary}")

        # Agent 1 revision based on critique
        print("\n--- [Agent 1 Revising Diagnosis Based on Critique...] ---")
        state.agent1_revision = revise_diagnosis(state, state.agent2_summary)
        print(f"Agent 1 Revision: {state.agent1_revision}")

        # Deterministic confidence after revision
        confidence = ConfidenceCalculator.stable_confidence(state)
        state.agent2_confidence = confidence

        # Optional clarification cycles if not sufficient and low confidence
        clarification_cycles = 0
        while confidence < 0.7 and clarification_cycles < CLARIFICATION_MAX_CYCLES and not data_sufficient(state):
            print(f"\n⚠️ Confidence ({confidence:.1%}) below threshold (Cycle {clarification_cycles + 1}/{CLARIFICATION_MAX_CYCLES}). Generating clarification questions...\n")
            clarification_questions = generate_clarification_questions(state)
            if clarification_questions:
                ask_generated_yes_no_questions(state, clarification_questions)
                state.agent2_summary, _ = create_final_summary(state) # Refresh critique
                state.agent1_revision = revise_diagnosis(state, state.agent2_summary)
                confidence = ConfidenceCalculator.stable_confidence(state)
                state.agent2_confidence = confidence
            else:
                break
            clarification_cycles += 1

        if confidence < 0.7:
            print("\n⚠️ Proceeding with best-effort diagnosis (confidence still low).\n")

        # Run contradiction detector
        contradictions = detect_contradictions(state)
        if contradictions:
            print("\n⚠️ Detected contradictions: " + "; ".join(contradictions))

        # Agent 3: Final Verification
        print("\n--- [Agent 3 Final Verification...] ---")
        final_output = final_verification(state)

        print("\n--- [FINAL CONSULTATION REPORT] ---\n")
        print(final_output)

    except DiagnosisRefusal as e:
        print(f"\n--- [EMERGENCY TRIAGE - SERVICE HALTED] ---")
        print(e.message)
        print("\nExiting diagnostic flow for your safety.")

if __name__ == "__main__":
    run_session()

if __name__ == "__main__":
    run_session()
