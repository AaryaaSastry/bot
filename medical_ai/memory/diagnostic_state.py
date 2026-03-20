from config import call_llm, SYSTEM_PROMPTS
from medical_ai.medical_framework import RedFlagDetector

class DiagnosticState:

    # OPQRST fields that must be collected
    OPQRST_FIELDS = ["onset", "provocation", "quality", "region", "severity", "time"]

    def __init__(self):
        self.symptoms_positive = []
        self.symptoms_negative = []
        self.question_history = []
        self.answers = []
        self.turn_count = 0
        self.question_count = 0
        self.chat_history = []  # Maintains complete conversation thread
        self.verification_feedbacks = [] # Store feedback from Agent 2
        self.cycle_count = 0 # Track transfers between Agent 1 and Agent 2
        self.extra_questions_asked = False # Flag for Agent 3's remedial questions
        self.is_final_phase = False # Track if we are in Agent 3's final check
        self.agent1_summary = ""
        self.agent2_summary = ""
        self.agent1_revision = ""  # revised diagnosis
        self.collected_fields = {} # Track OPQRST fields collected
        self.red_flags_found = [] # Store detected red flags
        self.opqrst_completed = False # Track if all OPQRST fields collected
        self.contradictions_found = False  # flag set by contradiction detector
        
        # New Ayurvedic Diagnostic Evidence
        self.tongue_description = ""
        self.digestion_status = ""
        self.sleep_pattern = ""
        self.energy_level = ""


    def record_interaction(self, question, answer):
        self.question_history.append(question)
        self.answers.append(answer)
        
        # Add to chat history
        self.chat_history.append({"role": "ai", "content": question})
        self.chat_history.append({"role": "user", "content": answer})

        # Check for red flags in user response
        red_flags = RedFlagDetector.check_red_flags(answer)
        if red_flags:
            self.red_flags_found.extend(red_flags)
        
        # Extract symptoms from all responses (not just initial)
        self._extract_symptoms(answer)
        
        self.turn_count += 1

    def get_conversation_history(self):
        """Returns formatted chat history as a string"""
        history = ""
        for i in range(len(self.question_history)):
            history += f"AI: {self.question_history[i]}\n"
            if i < len(self.answers):
                history += f"User: {self.answers[i]}\n"
        return history


    def get_chat_history_text(self):
        """Returns formatted chat history as a string"""
        history = []
        for message in self.chat_history:
            role = "AI" if message["role"] == "ai" else "User"
            history.append(f"{role}: {message['content']}")
        return "\n".join(history)


    def _extract_symptoms(self, text):
        # Check if text is primarily an introduction without symptoms
        intro_keywords = ['hi', 'hello', 'hey', 'i am', 'my name is', 'good morning', 'good afternoon', 'good evening']
        has_intro = any(keyword in text.lower() for keyword in intro_keywords)
        has_symptoms = any(word in text.lower() for word in ['pain', 'ache', 'fever', 'cough', 'sore', 'tired', 'nausea', 'vomit', 'headache', 'stomach', 'cold', 'flu', 'swelling', 'swollen'])

        if has_intro and not has_symptoms:
            return  # Skip symptom extraction for pure introductions

        prompt = f"""
Extract medical symptoms from the patient's text.

RULES:
- Only extract actual symptoms mentioned by the user
- Do not infer new symptoms
- Do not add explanations
- Use simple symptom names

TEXT:
{text}

OUTPUT:
List of symptoms, one per line.
"""

        response = call_llm(prompt, SYSTEM_PROMPTS["symptom_extractor"])
        symptoms = response.strip().split('\n')

        for symptom in symptoms:
            symptom = symptom.strip()
            if symptom:
                self.symptoms_positive.append(symptom)

    def record_opqrst_field(self, field, value):
        """Record a collected OPQRST field"""
        self.collected_fields[field] = value
        # Check if all OPQRST fields are collected
        self._check_opqrst_completion()

    def _check_opqrst_completion(self):
        """Check if all required OPQRST fields have been collected"""
        collected_count = sum(1 for f in self.OPQRST_FIELDS if self.collected_fields.get(f) is not None)
        if collected_count >= len(self.OPQRST_FIELDS):
            self.opqrst_completed = True

    def get_missing_opqrst_fields(self):
        """Return list of missing OPQRST fields"""
        return [f for f in self.OPQRST_FIELDS if self.collected_fields.get(f) is None]

    def is_opqrst_complete(self):
        """Check if all OPQRST fields are collected"""
        return self.opqrst_completed

    def get_opqrst_summary(self):
        """Get formatted OPQRST summary for verification"""
        summary = "\n=== OPQRST ASSESSMENT ===\n"
        field_names = {
            "onset": "Onset (When it started)",
            "provocation": "Provocation (What makes it worse/better)",
            "quality": "Quality (Description of symptom)",
            "region": "Region (Location)",
            "severity": "Severity (1-10 scale)",
            "time": "Time (Duration/Frequency)"
        }
        for field in self.OPQRST_FIELDS:
            value = self.collected_fields.get(field, "NOT COLLECTED")
            summary += f"{field_names.get(field, field)}: {value}\n"
        return summary 
