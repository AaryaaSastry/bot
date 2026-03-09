from medical_ai.memory.diagnostic_state import DiagnosticState
from medical_ai.agents.interviewer import ask_question, generate_simple_diagnosis
from medical_ai.agents.validator import verify_questions, create_final_summary
from medical_ai.agents.supervisor import final_verification
from config import MAX_QUESTIONS, SUPERVISOR_INTERVAL

def run_session():
    state = DiagnosticState()
    print("\n--- [AI Medical Diagnostic System Initialized] ---\n")

    # Agent 1 asks MAX_QUESTIONS questions
    for i in range(MAX_QUESTIONS):
        question = ask_question(state)
        print(f"Agent 1: {question}")
        answer = input("User: ")
        state.record_interaction(question, answer)
        
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
