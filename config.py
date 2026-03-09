import google.generativeai as genai
import ollama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API KEYS
GEMINI_API_KEY = "AIzaSyBjOwLujWyH-EFpQUgnxtX604wZmzRthAo"

# MODEL NAMES
GEMINI_MODEL = "gemma-3-27b-it"
OPENAI_MODEL = "gpt-oss:20b"  # This is now running via Ollama
OLLAMA_MODEL = "llama3"

# DEFAULT MODEL SELECTION
DEFAULT_MODEL = "gemini"  # Options: "gemini", "openai", "ollama"

# SETTINGS
MAX_QUESTIONS = 15
SUPERVISOR_INTERVAL = 5

# PROMPT TEMPLATES
SYSTEM_PROMPTS = {
    "interviewer": """You are Agent 1, a skilled, empathetic Ayurvedic medical interviewer. 
    Your goal is to gather comprehensive medical information from patients in a compassionate and professional manner. 
    Ask clear, concise questions that help identify symptoms, their severity, duration, and patterns. 
    Avoid medical jargon and explain any complex terms in simple language. 
    Build rapport with patients and create a comfortable environment for them to share information.""",
    
    "validator": """You are Agent 2, a senior medical supervisor with extensive experience in Ayurvedic diagnostics. 
    Your role is to review the interview process, evaluate the quality of questions, identify missing information, 
    and provide constructive feedback. Focus on clinical relevance, safety concerns, and diagnostic accuracy.""",
    
    "supervisor": """You are Agent 3, the final verifier and diagnostic expert. 
    Your responsibility is to analyze all collected information, verify the diagnosis, 
    and ensure the final report is accurate, comprehensive, and patient-friendly. 
    Prioritize patient safety and provide clear recommendations.""",
    
    "symptom_extractor": """You are a medical NLP expert specializing in symptom extraction. 
    Your task is to identify and extract all medical symptoms from patient descriptions. 
    Be precise and avoid including non-medical information. Extract symptoms in a structured format."""
}

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

def call_gemini(prompt, system_prompt=None, temperature=0.7):
    """Call Gemini API with enhanced configuration"""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = gemini_model.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2048
            )
        )
        
        if response and hasattr(response, 'text'):
            return response.text.strip()
        return ""
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return ""

def call_openai(prompt, system_prompt=None, temperature=0.7):
    """Call OpenAI-compatible API via Ollama with enhanced configuration"""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=OPENAI_MODEL,
            messages=messages,
            options={"temperature": temperature, "num_predict": 2048}
        )
        
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}")
        return ""

def call_ollama(prompt, system_prompt=None, temperature=0.7):
    """Call Ollama API with enhanced configuration"""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": temperature, "num_predict": 2048}
        )
        
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ollama API Error: {str(e)}")
        return ""

def call_llm(prompt, system_prompt=None, model=None, temperature=0.7):
    """Universal LLM call with fallback mechanism"""
    if model is None:
        model = DEFAULT_MODEL
    
    logger.info(f"Calling LLM: {model}")
    
    try:
        if model == "gemini":
            return call_gemini(prompt, system_prompt, temperature)
        elif model == "openai":
            return call_openai(prompt, system_prompt, temperature)
        elif model == "ollama":
            return call_ollama(prompt, system_prompt, temperature)
        else:
            raise ValueError(f"Unknown model: {model}")
    except Exception as e:
        logger.error(f"Primary model {model} failed: {str(e)}")
        
        # Fallback to other available models
        fallback_models = ["gemini", "openai", "ollama"]
        fallback_models.remove(model)
        
        for fallback in fallback_models:
            try:
                logger.info(f"Falling back to model: {fallback}")
                if fallback == "gemini":
                    return call_gemini(prompt, system_prompt, temperature)
                elif fallback == "openai":
                    return call_openai(prompt, system_prompt, temperature)
                elif fallback == "ollama":
                    return call_ollama(prompt, system_prompt, temperature)
            except Exception as fallback_e:
                logger.error(f"Fallback model {fallback} failed: {str(fallback_e)}")
        
        return "I'm sorry, but I'm having trouble processing your request right now. Please try again later."