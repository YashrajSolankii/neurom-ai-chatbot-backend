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

# ---------------- MODULE REGISTRY ----------------

MODULE_REGISTRY = {
    "breatheeasy_relax": {
        "module_name": "BreatheEasy - Relax",
        "category": "emotional_regulation"
    },
    "morning_meditation_guided": {
        "module_name": "Morning Meditation - Guided",
        "category": "emotional_regulation"
    },
    "anti_stress_music": {
        "module_name": "Anti Stress Music",
        "category": "music"
    },
    "power_nap_10": {
        "module_name": "Power Nap - 10 Minutes",
        "category": "sleep"
    },
    "journal": {
        "module_name": "Journal",
        "category": "reflection"
    },
    "gratitude_family": {
        "module_name": "Gratitude - Family",
        "category": "emotional_regulation"
    },
    "number_nest": {
        "module_name": "NumberNest",
        "category": "cognitive"
    },
    "ball_rush": {
        "module_name": "BallRush",
        "category": "high_energy"
    }
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

# ---------------- EMOTION DETECTION ----------------

def detect_emotion(text: str):
    text = text.lower()

    emotion_keywords = {
        "stress": ["stressed", "pressure", "overwhelmed", "exam", "deadline"],
        "anxiety": ["anxious", "nervous", "worried", "panic"],
        "sadness": ["sad", "down", "depressed", "unhappy"],
        "anger": ["angry", "frustrated", "irritated"],
        "burnout": ["exhausted", "burnt out", "drained", "tired of everything"],
        "loneliness": ["alone", "lonely", "isolated"],
        "positive": ["happy", "excited", "motivated"]
    }

    detected_emotion = "neutral"
    score = 0

    for emotion, keywords in emotion_keywords.items():
        for word in keywords:
            if word in text:
                detected_emotion = emotion
                score += 1

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

# ---------------- INTENT DETECTION ----------------

def detect_intent(text: str):
    text = text.lower()

    intent_map = {
        "meditation_request": ["meditate", "meditation", "mindfulness"],
        "breathing_request": ["breath", "breathing"],
        "music_request": ["music", "focus music"],
        "sleep_request": ["sleep", "nap", "power nap"],
        "journaling_request": ["journal", "write my thoughts"],
        "gratitude_request": ["grateful", "gratitude"],
        "affirmation_request": ["affirmation", "motivate me"],
        "cognitive_training": ["brain game", "improve focus", "memory"],
        "high_energy_game": ["game", "play something fun"],
        "knowledge_query": ["what is", "why does", "how does"]
    }

    for intent, keywords in intent_map.items():
        for word in keywords:
            if word in text:
                return intent

    return "emotional_regulation"

# ---------------- SMART ROUTING ----------------

def route_to_module(intent, emotion):

    if intent == "breathing_request":
        return "breatheeasy_relax"

    if intent == "meditation_request":
        return "morning_meditation_guided"

    if intent == "music_request":
        return "anti_stress_music"

    if intent == "sleep_request":
        return "power_nap_10"

    if intent == "journaling_request":
        return "journal"

    if intent == "gratitude_request":
        return "gratitude_family"

    if intent == "cognitive_training":
        return "number_nest"

    if intent == "high_energy_game":
        return "ball_rush"

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
        "want to die", "i don't want to live",
        "no point living", "harm myself",
        "self harm", "cut myself"
    ]

    medium_risk_keywords = [
        "life is meaningless",
        "i give up",
        "can't go on",
        "nothing matters",
        "i feel hopeless"
    ]

    risk_level = "none"
    matched_keywords = []

    for phrase in high_risk_keywords:
        if phrase in text:
            risk_level = "high"
            matched_keywords.append(phrase)

    if risk_level != "high":
        for phrase in medium_risk_keywords:
            if phrase in text:
                risk_level = "medium"
                matched_keywords.append(phrase)

    return {
        "risk_level": risk_level,
        "matched_keywords": matched_keywords
    }

# ---------------- HELPLINE ROUTING ----------------

def get_helpline(country_code: str):

    if country_code == "IN":
        return "Tele-MANAS (India National Mental Health Helpline): Dial 14416 (24/7, multilingual support)."

    elif country_code == "US":
        return "988 Suicide & Crisis Lifeline: Call or text 988 (24/7 support)."

    elif country_code == "UK":
        return "Samaritans UK: Call 116 123 (24/7 support)."

    else:
        return "Please contact your local emergency services or visit https://findahelpline.com to locate support in your country."

# ---------------- CRISIS RESPONSE TEMPLATE ----------------

def build_crisis_response(country_code: str = "IN"):

    helpline_info = get_helpline(country_code)

    return f"""
I'm really sorry that you're feeling this way. You are not alone, and help is available right now.

If you are in immediate danger, please call emergency services.

You can contact:
{helpline_info}

Talking to a trained professional can make a real difference.
Would you be willing to reach out to someone right now?
"""

# ---------------- INITIALIZATION ----------------

def initialize_resources():
    global LLM_INSTANCE, RETRIEVER_INSTANCE, RESOURCES_INITIALIZED

    if RESOURCES_INITIALIZED:
        return

    print("Initializing LLM and Retriever...")

    LLM_INSTANCE = Groq()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, CHROMA_PERSIST_DIRECTORY)

    if os.path.exists(db_path) and os.listdir(db_path):
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    else:
        all_pages = []
        for pdf in PDF_FILES_CONFIG:
            path = os.path.join(script_dir, pdf)
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                all_pages.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        docs = splitter.split_documents(all_pages)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=db_path
        )

    RETRIEVER_INSTANCE = vectorstore.as_retriever(search_kwargs={"k": 7})
    RESOURCES_INITIALIZED = True

# ---------------- MAIN RESPONSE FUNCTION ----------------

def generate_llm_response(user_query: str,
                          session_id: Optional[str] = None,
                          profile_data: Optional[dict] = None):

    if not RESOURCES_INITIALIZED:
        initialize_resources()

    if not session_id:
        session_id = generate_session_id()

    history = get_session_history(session_id)

    # ---- Crisis Detection ----
    crisis_data = detect_crisis(user_query)
    risk_level = crisis_data["risk_level"]

    if risk_level == "high":
        return {
            "session_id": session_id,
            "response": build_crisis_response("IN"),
            "emotion_detected": "crisis",
            "intent": "crisis",
            "confidence": 1.0,
            "intensity": "critical",
            "safe_mode": True
        }

    # ---- Emotion Detection ----
    emotion_data = detect_emotion(user_query)
    emotion = emotion_data["emotion"]
    intensity = emotion_data["intensity"]
    confidence = emotion_data["confidence"]

    # ---- Intent Detection ----
    intent = detect_intent(user_query)

    # ---- Routing ----
    primary_module_id = route_to_module(intent, emotion)
    module_data = MODULE_REGISTRY.get(primary_module_id)

    conversation_context = ""
    for msg in history:
        conversation_context += f"{msg['role'].capitalize()}: {msg['message']}\n"

    context_text = ""
    if intent == "knowledge_query":
        retrieved_docs = RETRIEVER_INSTANCE.invoke(user_query)
        context_text = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

    final_prompt = f"""
You are an emotionally intelligent assistant for the NeurOm mental wellness app.

Rules:
- Be supportive and emotionally aware.
- Keep response concise (3-4 lines).
- Encourage the recommended module strongly.
- Do NOT give outside advice.

Detected Emotion: {emotion}
Emotion Intensity: {intensity}
Detected Intent: {intent}
Recommended Module: {module_data['module_name']}

Conversation History:
{conversation_context}

Knowledge Base Context:
{context_text}

User Question:
{user_query}
"""

    # Build structured chat messages properly

    messages = [
        {
            "role": "system",
            "content": f"""
    You are an emotionally intelligent assistant inside the Neurom mental wellness app.

    Rules:
    - Always consider previous conversation history carefully.
    - If the user refers to something mentioned earlier (e.g., "they", "it", "that"), resolve it using past messages.
    - If the user asks about previously shared information, answer using conversation memory first.
    - Be supportive and emotionally aware.
    - Keep response concise (3-4 lines).
    - After answering, gently encourage the recommended module.
    - Do NOT give outside advice.
    - Keep user inside the app.
    """
        }
]

# Inject conversation history properly
    for msg in history:
        messages.append({
        "role": msg["role"],
        "content": msg["message"]
    })

# Inject emotion + intent context
    messages.append({
        "role": "system",
        "content": f"""
    Detected Emotion: {emotion}
    Emotion Intensity: {intensity}
    Detected Intent: {intent}
    Recommended Module: {module_data['module_name']}
    """
    })

# Add latest user query
    messages.append({
        "role": "user",
        "content": user_query
    })

    completion = LLM_INSTANCE.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2
    )

    answer = completion.choices[0].message.content

    update_session_history(session_id, "user", user_query)
    update_session_history(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "response": answer,
        "emotion_detected": emotion,
        "intent": intent,
        "confidence": confidence,
        "intensity": intensity,
        "safe_mode": False,
        "primary_recommendation": {
            "module_id": primary_module_id,
            "module_name": module_data["module_name"],
            "category": module_data["category"],
            "action": "open_module"
        }
    }

if not RESOURCES_INITIALIZED:
    initialize_resources()