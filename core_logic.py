# core_logic.py

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import uuid
from typing import Optional

print("core_logic.py: Loading environment variables...")
load_dotenv()

# ---------------- GLOBAL MEMORY STORE ----------------
SESSION_MEMORY = {}
MEMORY_WINDOW = 6

LLM_INSTANCE = None
RETRIEVER_INSTANCE = None
RESOURCES_INITIALIZED = False

PDF_FILES_CONFIG = [
    "CerboTech Chatbot doc (3).pdf",
    "The_Miracle_of_Mindfulness__An_Introductio_-_Thich_Nhat_Hanh.pdf",
    "zenmind.pdf",
    "Mindfulness_in_Plain_English.pdf",
    "Kathleen_McDonald_Robina_Courtin_How_to.pdf",
    "Daniel Goleman_ Richard J. Davidson - The Science of Meditation_ How to Change Your Brain, Mind and Body .pdf"
]

CHROMA_PERSIST_DIRECTORY = "chroma_db_api_neuroum"

# ---------------- MODULE REGISTRY (UPDATED) ----------------
MODULE_REGISTRY = {
    "breatheeasy_relax": {"module_name": "Breathing", "category": "emotional_regulation"},
    "morning_meditation_guided": {"module_name": "Morning Meditation", "category": "emotional_regulation"},
    "gratitude_family": {"module_name": "Gratitude", "category": "emotional_regulation"},
    "tratak_focus": {"module_name": "Tratak", "category": "emotional_regulation"},
    "power_nap_10": {"module_name": "Power Nap", "category": "sleep"},
    "journal": {"module_name": "Journaling", "category": "reflection"},
    "affirmation": {"module_name": "Affirmations", "category": "emotional_regulation"},
    "sherlock_mode": {"module_name": "Sherlock Mode", "category": "cognitive"},
    "number_nest": {"module_name": "Cognitive Games", "category": "cognitive"},
    "night_music": {"module_name": "Night Music", "category": "sleep"}
}

# ---------------- MEMORY HELPERS ----------------
def generate_session_id():
    return str(uuid.uuid4())

def get_session_history(session_id: str):
    return SESSION_MEMORY.get(session_id, [])

def update_session_history(session_id: str, role: str, message: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append({
        "role": role,
        "message": message
    })

    SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-MEMORY_WINDOW:]

# ---------------- EMOTION DETECTION (FIXED) ----------------
def detect_emotion(text: str):
    text = text.lower()

    emotion_keywords = {
        "stress": ["stressed", "pressure", "overwhelmed", "exam", "deadline"],
        "anxiety": ["anxious", "nervous", "worried", "panic"],
        "sadness": ["sad", "down", "depressed", "unhappy"],
        "anger": ["angry", "frustrated", "irritated"],
        "burnout": ["exhausted", "burnt out", "drained", "tired"],
        "loneliness": ["alone", "lonely", "isolated"],
        "positive": ["happy", "excited", "motivated"]
    }

    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        emotion_scores[emotion] = sum(1 for word in keywords if word in text)

    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    score = emotion_scores[detected_emotion]

    confidence = min(0.5 + (score * 0.1), 0.95)

    intensity = "low"
    if score >= 3:
        intensity = "high"
    elif score == 2:
        intensity = "medium"

    return {
        "emotion": detected_emotion,
        "confidence": round(confidence, 2),
        "intensity": intensity
    }

# ---------------- INTENT DETECTION (UPDATED) ----------------
def detect_intent(text: str):
    text = text.lower()

    intent_map = {
        "breathing_request": ["overwhelmed", "pressure", "tight chest"],
        "meditation_request": ["anxious", "panic", "uneasy"],
        "gratitude_request": ["empty", "sad", "low"],
        "tratak_request": ["angry", "frustrated", "rage"],
        "sleep_request": ["sleep", "insomnia", "restless"],
        "journaling_request": ["lonely", "alone", "isolated"],
        "affirmation_request": ["confidence", "motivation", "self worth"],
        "sherlock_request": ["overthinking", "looping", "thinking too much"],
        "cognitive_training": ["focus", "concentration"],
        "music_request": ["sleep music", "night"],
        "knowledge_query": ["what is", "why", "how"]
    }

    for intent, keywords in intent_map.items():
        if any(word in text for word in keywords):
            return intent

    return "emotional_regulation"

# ---------------- ROUTING ----------------
def route_to_module(intent, emotion):

    if intent == "breathing_request":
        return "breatheeasy_relax"
    if intent == "meditation_request":
        return "morning_meditation_guided"
    if intent == "gratitude_request":
        return "gratitude_family"
    if intent == "tratak_request":
        return "tratak_focus"
    if intent == "sleep_request":
        return "power_nap_10"
    if intent == "journaling_request":
        return "journal"
    if intent == "affirmation_request":
        return "affirmation"
    if intent == "sherlock_request":
        return "sherlock_mode"
    if intent == "cognitive_training":
        return "number_nest"
    if intent == "music_request":
        return "night_music"

    if emotion in ["stress", "anxiety", "burnout"]:
        return "breatheeasy_relax"
    if emotion == "sadness":
        return "gratitude_family"

    return "morning_meditation_guided"

# ---------------- CRISIS DETECTION ----------------
def detect_crisis(text: str):
    text = text.lower()

    high_risk_keywords = [
        "kill myself", "suicide", "end my life",
        "want to die", "die", "feel to die",
        "i don't want to live", "harm myself",
        "self harm", "cut myself"
    ]

    medium_risk_keywords = [
        "life is meaningless",
        "i give up",
        "can't go on",
        "nothing matters",
        "hopeless",
        "tired of everything"
    ]

    # High risk detection
    for phrase in high_risk_keywords:
        if phrase in text:
            return {"risk_level": "high", "matched_keywords": [phrase]}

    # Medium risk detection
    for phrase in medium_risk_keywords:
        if phrase in text:
            return {"risk_level": "medium", "matched_keywords": [phrase]}

    return {"risk_level": "none", "matched_keywords": []}

    # Smart pattern detection
    if "die" in text and ("feel" in text or "want" in text):
        return {"risk_level": "high", "matched_keywords": ["implicit_suicidal_intent"]}

# ---------------- INITIALIZATION ----------------
def initialize_resources():
    global LLM_INSTANCE, RETRIEVER_INSTANCE, RESOURCES_INITIALIZED

    if RESOURCES_INITIALIZED:
        return

    LLM_INSTANCE = Groq()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    RETRIEVER_INSTANCE = vectorstore.as_retriever(search_kwargs={"k": 7})
    RESOURCES_INITIALIZED = True

# ---------------- MAIN FUNCTION ----------------
def generate_llm_response(user_query: str,
                          session_id: Optional[str] = None,
                          profile_data: Optional[dict] = None):

    if not RESOURCES_INITIALIZED:
        initialize_resources()

    if not session_id:
        session_id = generate_session_id()

    history = get_session_history(session_id)

    # Crisis
    crisis_data = detect_crisis(user_query)
    if crisis_data["risk_level"] == "high":

        update_session_history(session_id, "user", user_query)
        update_session_history(session_id, "assistant", "CRISIS")

        return {
            "session_id": session_id,
            "response": "You are not alone. Please reach out to Tele-MANAS at 14416.",
            "emotion_detected": "crisis",
            "intent": "crisis",
            "confidence": 1.0,
            "intensity": "critical",
            "safe_mode": True
        }

    emotion_data = detect_emotion(user_query)
    intent = detect_intent(user_query)

    module_id = route_to_module(intent, emotion_data["emotion"])
    module_data = MODULE_REGISTRY[module_id]

    # ---------------- PROMPT (RESTORED STYLE) ----------------
    messages = [
        {
            "role": "system",
            "content": """
You are an emotionally intelligent assistant for the NeurOm mental wellness app.

STRICT RULES (MUST FOLLOW):
- ONLY recommend modules from this list:
  Breathing, Morning Meditation, Gratitude, Tratak, Power Nap, Journaling, Affirmations, Sherlock Mode, Cognitive Games, Night Music
- DO NOT mention any other activity, feature, or technique outside this list
- DO NOT invent or suggest new modules
- DO NOT use general knowledge to suggest features

- Be supportive, natural, and conversational.
- Give meaningful responses (not short robotic replies).
- You can comfort, guide, and explain briefly.
- DO NOT explain navigation or app paths.
"""
        }
    ]

    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["message"]
        })

    messages.append({
        "role": "system",
        "content": f"""
Detected Emotion: {emotion_data['emotion']}
Detected Intent: {intent}
Recommended Module: {module_data['module_name']}
"""
    })

    messages.append({
        "role": "user",
        "content": user_query
    })

    completion = LLM_INSTANCE.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3
    )

    answer = completion.choices[0].message.content

    update_session_history(session_id, "user", user_query)
    update_session_history(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "response": answer,
        "emotion_detected": emotion_data["emotion"],
        "intent": intent,
        "confidence": emotion_data["confidence"],
        "intensity": emotion_data["intensity"],
        "safe_mode": False,
        "primary_recommendation": {
            "module_id": module_id,
            "module_name": module_data["module_name"],
            "category": module_data["category"],
            "action": "open_module"
        }
    }

if not RESOURCES_INITIALIZED:
    initialize_resources()