# Comprehensive Medical Question Framework
# Implements OPQRST, Two-Phase System, Red-Flag Detection, Symptom Scoring

from config import call_llm, SYSTEM_PROMPTS

# ============================================================================
# 1. STRUCTURED MEDICAL QUESTION FRAMEWORK (OPQRST)
# ============================================================================

class OPQRSTQuestionFramework:
    """Medical symptom assessment using OPQRST framework"""
    
    @staticmethod
    def get_next_question(symptom_type, collected_fields):
        """Return next question based on OPQRST framework"""
        
        questions = {
            "O": {
                "question": "When did this symptom start?",
                "field": "onset",
                "alternatives": [
                    "When did the pain/start?",
                    "How long ago did this begin?"
                ]
            },
            "P": {
                "question": "What makes it worse or better?",
                "field": "provocation",
                "alternatives": [
                    "Does anything make it better?",
                    "What triggers the symptoms?"
                ]
            },
            "Q": {
                "question": "How would you describe the pain? (throbbing, sharp, dull, burning)",
                "field": "quality",
                "alternatives": [
                    "Is it sharp, dull, or throbbing?",
                    "What does the pain feel like?"
                ]
            },
            "R": {
                "question": "Where exactly is the symptom located?",
                "field": "region",
                "alternatives": [
                    "Where do you feel it?",
                    "Where is the pain?"
                ]
            },
            "S": {
                "question": "How severe is it on a scale of 1-10?",
                "field": "severity",
                "alternatives": [
                    "Rate the pain 1-10",
                    "How bad is it?"
                ]
            },
            "T": {
                "question": "How long does it last?",
                "field": "time",
                "alternatives": [
                    "How often does it occur?",
                    "Does it happen daily or occasionally?"
                ]
            }
        }
        
        # Find first missing field
        for key in ["O", "P", "Q", "R", "S", "T"]:
            if collected_fields.get(questions[key]["field"]) is None:
                return questions[key]
        
        return None


# ============================================================================
# 2. RED-FLAG SAFETY DETECTION
# ============================================================================

class RedFlagDetector:
    """Detect dangerous medical symptoms requiring immediate attention"""
    
    # Common red flags for various symptoms
    RED_FLAG_PATTERNS = {
        "headache": [
            "worst headache ever",
            "sudden severe",
            "fever with headache",
            "vision loss",
            "confusion",
            "vomiting",
            "neck stiffness",
            "rash",
            "seizure"
        ],
        "chest_pain": [
            "chest pain",
            "pressure in chest",
            "arm pain",
            "jaw pain",
            "shortness of breath",
            "sweating",
            "nausea"
        ],
        "abdominal": [
            "severe pain",
            "blood",
            "vomiting blood",
            "high fever",
            "cannot pass stool",
            "pregnant"
        ],
        "general": [
            "bleeding",
            "difficulty breathing",
            "loss of consciousness",
            "slurred speech",
            "one sided weakness",
            "severe allergic reaction"
        ]
    }
    
    @staticmethod
    def check_red_flags(symptoms_text, symptom_type="general"):
        """Check for red flag symptoms"""
        text_lower = symptoms_text.lower()
        
        # Check specific category
        patterns = RedFlagDetector.RED_FLAG_PATTERNS.get(
            symptom_type, 
            RedFlagDetector.RED_FLAG_PATTERNS["general"]
        )
        
        found_flags = []
        for pattern in patterns:
            if pattern in text_lower:
                found_flags.append(pattern)
        
        return found_flags
    
    @staticmethod
    def get_red_flag_warning(found_flags):
        """Generate warning message for red flags"""
        if not found_flags:
            return None
            
        return (
            "⚠️ SAFETY WARNING: Your symptoms may require immediate medical attention. "
            "Please consult a healthcare professional urgently if you experience: "
            f"{', '.join(found_flags).upper()}"
        )


# ============================================================================
# 3. TWO-PHASE SYSTEM - Data Collection vs Diagnosis
# ============================================================================

class TwoPhaseSystem:
    """Separates data collection from diagnosis"""
    
    # Minimum required fields for different symptom types
    MINIMUM_FIELDS = {
        "pain": ["location", "onset", "severity", "provocation", "quality"],
        "digestive": ["onset", "frequency", "associated_symptoms"],
        "respiratory": ["onset", "severity", "triggers", "associated_symptoms"],
        "general": ["onset", "severity", "associated_symptoms"]
    }
    
    @staticmethod
    def get_symptom_category(symptoms):
        """Determine symptom category for field requirements"""
        text = " ".join(symptoms).lower()
        
        if any(w in text for w in ["headache", "pain", "ache", "stomach"]):
            return "pain"
        elif any(w in text for w in ["digestion", "stomach", "bowel", "nausea"]):
            return "digestive"
        elif any(w in text for w in ["cough", "breathing", "breath", "cold"]):
            return "respiratory"
        else:
            return "general"
    
    @staticmethod
    def is_data_sufficient(collected_fields, symptom_category):
        """Check if enough data has been collected"""
        required = TwoPhaseSystem.MINIMUM_FIELDS.get(symptom_category, ["onset", "severity"])
        collected = sum(1 for f in required if collected_fields.get(f) is not None)
        
        # Require at least 60% of fields
        threshold = len(required) * 0.6
        return collected >= threshold


# ============================================================================
# 4. SYMPTOM SCORING SYSTEM - Dosha Assessment
# ============================================================================

class SymptomScorer:
    """Score symptoms to determine dominant dosha"""
    
    # Symptom to dosha mapping with weights
    DOSHA_SCORES = {
        "vata": {
            "dry": 2, "cracking": 2, "cold": 1, "restless": 2, 
            "anxious": 2, "bloating": 2, "constipation": 2,
            "light": 1, "thin": 1, "trembling": 2, "nervous": 2
        },
        "pitta": {
            "heat": 2, "warm": 2, "burning": 3, "inflamed": 2,
            "irritable": 2, "angry": 2, "acid": 2, "heartburn": 2,
            "sharp": 1, "intense": 2, "yellow": 1, "red": 1
        },
        "kapha": {
            "heavy": 2, "cold": 1, "slow": 2, "congested": 2,
            "mucus": 2, "swelling": 2, "oily": 1, "dull": 2,
            "sleepy": 2, "lazy": 2, "pale": 1, "wet": 1
        }
    }
    
    @staticmethod
    def calculate_dosha_scores(symptoms):
        """Calculate dosha dominance scores"""
        scores = {"vata": 0, "pitta": 0, "kapha": 0}
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            
            for dosha, symptom_weights in SymptomScorer.DOSHA_SCORES.items():
                for key, weight in symptom_weights.items():
                    if key in symptom_lower:
                        scores[dosha] += weight
        
        return scores
    
    @staticmethod
    def get_dominant_dosha(scores):
        """Get dominant dosha from scores"""
        if not scores or sum(scores.values()) == 0:
            return "unknown"
        
        return max(scores, key=scores.get)


# ============================================================================
# 5. PROBABILISTIC DIAGNOSIS
# ============================================================================

class ProbabilisticDiagnosis:
    """Generate differential diagnoses with probabilities"""
    
    # Condition indicators with symptoms and probabilities
    CONDITION_RULES = {
        "migraine": {
            "indicators": ["throbbing", "temple", "light sensitivity", "nausea", "sun", "bright"],
            "base_probability": 0.30
        },
        "tension_headache": {
            "indicators": ["dull", "pressure", "stress", "neck", "tightness"],
            "base_probability": 0.25
        },
        "heat_headache": {
            "indicators": ["heat", "sun", "warm", "dehydration", "summer"],
            "base_probability": 0.20
        },
        "sinus_headache": {
            "indicators": ["sinus", "forehead", "congestion", "mucus", "pressure"],
            "base_probability": 0.15
        }
    }
    
    @staticmethod
    def generate_differential_diagnosis(symptoms):
        """Generate differential diagnoses with probabilities"""
        diagnoses = []
        
        for condition, rules in ProbabilisticDiagnosis.CONDITION_RULES.items():
            match_count = 0
            for indicator in rules["indicators"]:
                if any(indicator in s.lower() for s in symptoms):
                    match_count += 1
            
            # Calculate probability based on matches
            if match_count > 0:
                probability = min(rules["base_probability"] + (match_count * 0.15), 0.85)
                diagnoses.append({
                    "condition": condition,
                    "probability": probability,
                    "match_count": match_count
                })
        
        # Sort by probability
        diagnoses.sort(key=lambda x: x["probability"], reverse=True)
        
        # Normalize probabilities
        total = sum(d["probability"] for d in diagnoses)
        if total > 0:
            for d in diagnoses:
                d["probability"] = round(d["probability"] / total, 2)
        
        return diagnoses[:3]  # Return top 3


# ============================================================================
# 6. CONFIDENCE SCORE CALCULATION
# ============================================================================

class ConfidenceCalculator:
    """Calculate honest confidence scores based on data completeness"""
    
    @staticmethod
    def calculate_confidence(model_confidence, data_completeness):
        """
        Calculate honest confidence score
        confidence = model_confidence * data_completeness
        """
        return round(model_confidence * data_completeness, 2)
    
    @staticmethod
    def calculate_data_completeness(collected_fields, required_fields):
        """Calculate percentage of required fields collected"""
        if not required_fields:
            return 0.5
        
        collected = sum(1 for f in required_fields if collected_fields.get(f) is not None)
        return collected / len(required_fields)

    @staticmethod
    def stable_confidence(state):
        """
        Deterministic confidence (0-1) using observable evidence only.
        Buckets (0.20 each):
        - OPQRST complete
        - Trigger/provocation identified
        - Symptom pattern match (>=2 symptoms)
        - Agent 1 & Agent 2 produced summaries (agreement proxy)
        - No contradictions flagged
        """
        score = 0.0

        # OPQRST completeness
        opqrst_fields = getattr(state, "OPQRST_FIELDS", ["onset", "provocation", "quality", "region", "severity", "time"])
        filled = sum(1 for f in opqrst_fields if getattr(state, "collected_fields", {}).get(f))
        if filled == len(opqrst_fields):
            score += 0.20

        # Trigger / provocation
        if getattr(state, "collected_fields", {}).get("provocation"):
            score += 0.20

        # Symptom pattern (at least two symptoms)
        if len(getattr(state, "symptoms_positive", [])) >= 2:
            score += 0.20

        # Agent summaries present (proxy for agreement)
        if getattr(state, "agent1_summary", "") and getattr(state, "agent2_summary", ""):
            score += 0.20

        # No contradictions flagged
        contradictions = getattr(state, "contradictions_found", False)
        if not contradictions:
            score += 0.20

        return round(min(score, 1.0), 2)


# ============================================================================
# 7. KNOWLEDGE LAYER - Condition Rules
# ============================================================================

class KnowledgeLayer:
    """Condition-specific knowledge rules"""
    
    # Migraine indicators
    MIGRAINE_INDICATORS = {
        "symptoms": ["throbbing pain", "temple location", "light sensitivity", "nausea", "worse in sunlight"],
        "triggers": ["sun", "stress", "lack of sleep", "certain foods", "bright light"],
        "questions": [
            "Where exactly is the pain?",
            "Is the pain throbbing or constant?",
            "Does light bother you?",
            "Do you feel nauseous?",
            "What makes it worse?"
        ]
    }
    
    # Tension headache indicators
    TENSION_HEADACHE_INDICATORS = {
        "symptoms": ["dull pain", "pressure", "tightness", "neck pain", "worse with stress"],
        "triggers": ["stress", "poor posture", "lack of sleep", "eye strain"],
        "questions": [
            "Do you feel stressed?",
            "Is your neck tight?",
            "How long does the pain last?",
            "Does it improve with rest?"
        ]
    }
    
    # Heat headache indicators
    HEAT_HEADACHE_INDICATORS = {
        "symptoms": ["worse in heat", "dehydration", "sun exposure"],
        "triggers": ["sun", "heat", "dehydration", "hot weather"],
        "questions": [
            "Does heat make it worse?",
            "Do you drink enough water?",
            "Are you exposed to sun?"
        ]
    }
    
    @staticmethod
    def get_relevant_questions(symptom_type):
        """Get condition-specific questions"""
        rules = getattr(KnowledgeLayer, f"{symptom_type.upper()}_INDICATORS", None)
        if rules:
            return rules.get("questions", [])
        return []


# ============================================================================
# 8. DIAGNOSIS REFUSAL MODE
# ============================================================================

class DiagnosisRefusal:
    """Handle insufficient information scenarios"""
    
    @staticmethod
    def should_refuse_diagnosis(confidence_threshold, current_confidence, data_completeness):
        """Determine if diagnosis should be refused"""
        return current_confidence < confidence_threshold or data_completeness < 0.5
    
    @staticmethod
    def get_refusal_message():
        """Message when diagnosis cannot be made"""
        return (
            "Insufficient information for a confident diagnosis. "
            "Please answer a few more questions to help me understand your symptoms better."
        )


# ============================================================================
# 9. IMPROVED FINAL REPORT STRUCTURE
# ============================================================================

class ReportFormatter:
    """Format final diagnostic reports professionally"""
    
    @staticmethod
    def format_report(symptoms, diagnoses, confidence, dosha_analysis, recommendations):
        """Generate professional medical report"""
        
        report = """
═══════════════════════════════════════════════════════════
              AYURVEDIC DIAGNOSTIC REPORT
═══════════════════════════════════════════════════════════

PATIENT SYMPTOMS
────────────────
"""
        for symptom in symptoms:
            report += f"• {symptom}\n"
        
        report += """
POSSIBLE CONDITIONS
───────────────────
"""
        for i, diag in enumerate(diagnoses, 1):
            prob = int(diag.get("probability", 0) * 100)
            report += f"{i}. {diag.get('condition', 'Unknown').replace('_', ' ').title()} ({prob}%)\n"
        
        report += f"""
CONFIDENCE LEVEL
────────────────
{int(confidence * 100)}% - {"High" if confidence > 0.7 else "Moderate" if confidence > 0.4 else "Low"}

AYURVEDIC ANALYSIS
─────────────────
{dosha_analysis}

RECOMMENDED NEXT STEPS
──────────────────────
"""
        for rec in recommendations:
            report += f"• {rec}\n"
        
        report += """
═══════════════════════════════════════════════════════════
DISCLAIMER: This is an AI-assisted preliminary assessment.
Please consult a qualified healthcare professional for proper diagnosis.
═══════════════════════════════════════════════════════════
"""
        return report
