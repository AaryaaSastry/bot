from config import call_gemini

class DiagnosticState:

    def __init__(self):
        self.symptoms_positive = []
        self.symptoms_negative = []
        self.question_history = []
        self.answers = []
        self.turn_count = 0
        self.chat_history = []  # Maintains complete conversation thread
        self.verification_feedbacks = [] # Store feedback from Agent 2
        self.cycle_count = 0 # Track transfers between Agent 1 and Agent 2
        self.extra_questions_asked = False # Flag for Agent 3's remedial questions
        self.is_final_phase = False # Track if we are in Agent 3's final check
        self.agent1_summary = ""
        self.agent2_summary = ""


    def record_interaction(self, question, answer):
        self.question_history.append(question)
        self.answers.append(answer)
        
        # Add to chat history
        self.chat_history.append({"role": "ai", "content": question})
        self.chat_history.append({"role": "user", "content": answer})

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
        Extract all medical symptoms from the following text.
        Return only a list of symptoms, one per line.
        Do not include any explanations or extra text.

        Text: {text}
        """

        response = call_gemini(prompt)
        symptoms = response.strip().split('\n')

        for symptom in symptoms:
            symptom = symptom.strip()
            if symptom:
                self.symptoms_positive.append(symptom) 