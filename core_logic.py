from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import uuid
from typing import Optional
from transformers import pipeline
import torch

# ---------------- FIX: load_dotenv FIRST before anything else ----------------
print("core_logic.py: Loading environment variables...")
load_dotenv()

# ---------------- DEVICE SETUP ----------------
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"Emotion model running on: {'GPU' if DEVICE == 0 else 'CPU'}")

# ---------------- LOAD MODEL ONCE ----------------
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=DEVICE
)
# ---------------- INTENT CLASSIFIER ----------------
intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE
)



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
    "sherlock_holmes": {"module_name": "Sherlock holmes", "category": "cognitive"},
    "cognitive_games": {"module_name": "Cognitive Games", "category": "cognitive"},
    "night_music": {"module_name": "Night Music", "category": "sleep"},

    #Real cognitive games from NeurOm app
    "mindflip": {"module_name": "MindFlip", "category": "cognitive"},
    "number_nest": {"module_name": "NumberNest", "category": "cognitive"},
    "wordhunt": {"module_name": "WordHunt", "category": "cognitive"},
    "alphaquest": {"module_name": "AlphaQuest", "category": "cognitive"},
    "percentpro": {"module_name": "PercentPro", "category": "cognitive"},
    "numberstorm": {"module_name": "NumberStorm", "category": "cognitive"},
    "ballrush": {"module_name": "BallRush", "category": "cognitive"},
    "rushhour": {"module_name": "RushHour", "category": "cognitive"},
    "stackup": {"module_name": "StackUp", "category": "cognitive"},
    "brickbreaker": {"module_name": "BrickBreaker", "category": "cognitive"},

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


# ---------------- EMOTION DETECTION (FINAL HYBRID SYSTEM) ----------------
import re

def normalize_text(text: str):
    text = text.lower()

    # Normalize apostrophes and symbols
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_emotion(text: str):
    try:
        clean_text = normalize_text(text)

        # ---------------- AI MODEL ----------------
        results = emotion_classifier(clean_text)[0]

        label_mapping = {
            "anger": "anger",
            "disgust": "anger",
            "fear": "anxiety",
            "joy": "positive",
            "neutral": "neutral",
            "sadness": "sadness",
            "surprise": "anxiety"
        }

        best = max(results, key=lambda x: x["score"])
        mapped_emotion = label_mapping.get(best["label"], "neutral")
        model_confidence = float(best["score"])

        
        negative_signals = [
        "nothing", "no one", "never", "not working",
        "not going", "wrong", "off", "bad", "stuck"
        ]

        if any(word in clean_text for word in negative_signals):
            if mapped_emotion in ["neutral", "positive"]:
                mapped_emotion = "sadness"
                model_confidence = max(model_confidence, 0.65)

        # ---------------- SEMANTIC PATTERNS ----------------
        sadness_patterns = [
            "feel nothing", "going through", "empty", "numb",
            "no purpose", "pointless", "lost interest"
        ]

        anxiety_patterns = [
            "mind won't stop", "can't stop thinking",
            "thoughts keep", "over and over",
            "replaying", "won't slow down"
        ]

        burnout_patterns = [
            "always tired", "no energy",
            "drained", "exhausted", "burnt out"
        ]

        pattern_emotion = None
        pattern_strength = 0

        if any(p in clean_text for p in sadness_patterns):
            pattern_emotion = "sadness"
            pattern_strength = 0.75

        elif any(p in clean_text for p in anxiety_patterns):
            pattern_emotion = "anxiety"
            pattern_strength = 0.75

        elif any(p in clean_text for p in burnout_patterns):
            pattern_emotion = "burnout"
            pattern_strength = 0.75

        # ---------------- DECISION LOGIC ----------------
        if model_confidence >= 0.65:
            final_emotion = mapped_emotion
            confidence = model_confidence

        elif pattern_emotion:
            final_emotion = pattern_emotion
            confidence = max(model_confidence, pattern_strength)

        else:
            final_emotion = mapped_emotion
            confidence = model_confidence

        confidence = round(confidence, 2)

        # ---------------- INTENSITY ----------------
        if confidence >= 0.75:
            intensity = "high"
        elif confidence >= 0.5:
            intensity = "medium"
        else:
            intensity = "low"

        return {
            "emotion": final_emotion,
            "confidence": confidence,
            "intensity": intensity
        }

    except Exception as e:
        print("Emotion detection error:", e)
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "intensity": "low"
        }

# ---------------- INTENT DETECTION (AI-IMPROVED) ----------------

def detect_intent(text: str):
    clean_text = normalize_text(text)

    candidate_labels = [
        "user needs help calming down",
        "user is feeling anxious or overwhelmed",
        "user feels sad or emotionally low",
        "user wants to sleep or relax",
        "user feels lonely or isolated",
        "user wants motivation or confidence",
        "user is overthinking or stuck in thoughts",
        "user wants to improve focus",
        "user wants relaxing music",
        "user is asking for knowledge or explanation"
    ]

    try:
        result = intent_classifier(
            clean_text,
            candidate_labels,
            multi_label=False
        )

        top_label = result["labels"][0]
        score = result["scores"][0]

        # ---------------- LABEL MAPPING ----------------
        mapping = {
            "user needs help calming down": "breathing_request",
            "user is feeling anxious or overwhelmed": "meditation_request",
            "user feels sad or emotionally low": "sherlock_request",
            "user wants to sleep or relax": "sleep_request",
            "user feels lonely or isolated": "journaling_request",
            "user wants motivation or confidence": "affirmation_request",
            "user is overthinking or stuck in thoughts": "sherlock_request",
            "user wants to improve focus": "cognitive_training",
            "user wants relaxing music": "music_request",
            "user is asking for knowledge or explanation": "knowledge_query"
        }

        predicted_intent = mapping.get(top_label, "emotional_regulation")

        # ---------------- CONFIDENCE FALLBACK ----------------
        if score < 0.4:
            return "emotional_regulation"

        return predicted_intent

    except Exception as e:
        print("Intent detection error:", e)
        return "emotional_regulation"

# ---------------- ROUTING ----------------
def route_to_module(intent: str, emotion: str, user_query: str) -> str:
    text = user_query.lower()

    # ── INTENT-BASED ROUTING (highest priority) ──
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
        return "sherlock_holmes"
    if intent == "cognitive_training":
        return "cognitive_games"
    if intent == "music_request":
        return "night_music"

    

    # Breathing — stress, pressure, overwhelmed, panic
    stress_keywords = [
        "stressed", "stress", "overwhelmed", "pressure", "panic",
        "tight chest", "suffocated", "suffocating", "overloaded",
        "too much", "can't cope", "breathless", "urgent", "deadline"
    ]
    if any(w in text for w in stress_keywords):
        return "breatheeasy_relax"

    # Morning Meditation — anxiety, fear, nervousness, unease
    anxiety_keywords = [
        "anxious", "anxiety", "nervous", "scared", "fear", "worried",
        "worrying", "dread", "uneasy", "panic attack", "overthinking",
        "hyper", "alert", "on edge", "restless mind", "tense"
    ]
    if any(w in text for w in anxiety_keywords):
        return "morning_meditation_guided"

    # Night Music — sleep issues
    sleep_keywords = [
        "can't sleep", "insomnia", "sleep", "sleepless", "awake at night",
        "night thoughts", "racing thoughts at night", "restless night",
        "tired but can't sleep", "sleep problem", "difficulty sleeping"
    ]
    if any(w in text for w in sleep_keywords):
        return "night_music"

    # Power Nap — burnout, exhaustion, drained
    burnout_keywords = [
        "burnout", "burnt out", "exhausted", "drained", "no energy",
        "mentally tired", "fatigue", "worn out", "no motivation",
        "sluggish", "lethargic", "completely tired", "energy crash"
    ]
    if any(w in text for w in burnout_keywords):
        return "power_nap_10"

    # Journaling — loneliness, isolation, suppressed feelings
    loneliness_keywords = [
        "lonely", "alone", "isolated", "no one", "no friends",
        "no one understands", "feel invisible", "disconnected",
        "left out", "abandoned", "no one cares", "feel empty inside",
        "no one to talk to", "suppressed", "unheard"
    ]
    if any(w in text for w in loneliness_keywords):
        return "journal"

    # Tratak — anger, frustration, irritation
    anger_keywords = [
        "angry", "anger", "furious", "irritated", "frustrated",
        "rage", "mad", "annoyed", "aggressive", "hostile",
        "irritation", "burst", "explosive", "resentment", "bitter"
    ]
    if any(w in text for w in anger_keywords):
        return "tratak_focus"

    # Affirmations — low confidence, negative self-talk, positive intent
    affirmation_keywords = [
        "not good enough", "worthless", "hate myself", "confidence",
        "self doubt", "insecure", "i can't do anything", "failure",
        "loser", "useless", "no self worth", "i am bad", "not capable"
    ]
    if any(w in text for w in affirmation_keywords):
        return "affirmation"

    # Sherlock — overthinking, mental loops, analysis paralysis
    overthinking_keywords = [
        "overthinking", "can't stop thinking", "mind won't stop",
        "thoughts keep", "over and over", "replaying", "mental loop",
        "can't decide", "analysis paralysis", "stuck in my head",
        "circular thoughts", "thinking too much"
    ]
    if any(w in text for w in overthinking_keywords):
        return "sherlock_holmes"

    # Gratitude — sadness, hopelessness, low mood
    sadness_keywords = [
        "sad", "sadness", "depressed", "hopeless", "empty", "numb",
        "no purpose", "pointless", "lost interest", "nothing matters",
        "feel nothing", "meaningless", "joyless", "melancholy",
        "heartbroken", "grief", "feel down", "low mood"
    ]
    if any(w in text for w in sadness_keywords):
        return "gratitude_family"

    # ── EMOTION-BASED FALLBACK ──
    emotion_map = {
        "anxiety":  "morning_meditation_guided",
        "stress":   "breatheeasy_relax",
        "burnout":  "power_nap_10",
        "sadness":  "gratitude_family",
        "anger":    "tratak_focus",
        "positive": "affirmation",
        "neutral":  "morning_meditation_guided",
    }
    return emotion_map.get(emotion, "morning_meditation_guided")

# ---------------- CRISIS DETECTION (Updated) ----------------
def detect_crisis(text: str):
    text = text.lower()

    high_risk_keywords = [
        "kill myself", "suicide", "end my life",
        "want to die", "i want to die",
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

    # Smart implicit detection (FIXED POSITION)
    if "die" in text and ("feel" in text or "want" in text):
        return {"risk_level": "high", "matched_keywords": ["implicit_suicidal_intent"]}

    # Medium risk detection
    for phrase in medium_risk_keywords:
        if phrase in text:
            return {"risk_level": "medium", "matched_keywords": [phrase]}

    return {"risk_level": "none", "matched_keywords": []}

# ---------------- GREETING / SMALL TALK DETECTION ----------------
GREETING_PATTERNS = [
    "hi", "hey", "hello", "hii", "heyy", "heyyy", "sup", "what's up",
    "whats up", "yo", "good morning", "good evening", "good afternoon",
    "good night", "howdy", "greetings", "namaste", "hola",
]

SMALL_TALK_PATTERNS = [
    "how are you", "how r u", "how are u", "what are you", "who are you",
    "tell me about yourself", "what can you do", "what do you do",
    "are you a bot", "are you ai", "are you real", "okay", "ok", "fine",
    "alright", "sure", "thanks", "thank you", "cool", "nice", "great",
    "awesome", "bye", "goodbye", "see you", "take care",
]

def is_greeting_or_small_talk(text: str) -> bool:
    clean = text.lower().strip().rstrip("!?.").strip()
    # Exact match for very short greetings
    if clean in GREETING_PATTERNS:
        return True
    # Short message (under 4 words) that starts with a greeting word
    words = clean.split()
    if len(words) <= 3 and any(clean.startswith(g) for g in GREETING_PATTERNS):
        return True
    # Small talk patterns
    if any(pattern in clean for pattern in SMALL_TALK_PATTERNS):
        return True
    return False

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
    # ── GREETING / SMALL TALK — no module recommended ──
    if is_greeting_or_small_talk(user_query):
        update_session_history(session_id, "user", user_query)

        greeting_response = LLM_INSTANCE.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are NeurOm, a warm and empathetic mental wellness companion.
The user has sent a greeting or casual message. Respond warmly and naturally.
Ask ONE gentle, open-ended follow-up question to understand how they are feeling today.
Do NOT recommend any modules or activities yet.
Keep response short — 2 to 3 sentences maximum."""
                },
                {"role": "user", "content": user_query}
            ],
            temperature=0.5
        )

        answer = greeting_response.choices[0].message.content
        update_session_history(session_id, "assistant", answer)

        return {
            "session_id": session_id,
            "response": answer,
            "emotion_detected": "neutral",
            "intent": "greeting",
            "confidence": 1.0,
            "intensity": "low",
            "safe_mode": False,
            "rag_used": False,
            "primary_recommendation": None,   # ← No module card shown
        }

    emotion_data = detect_emotion(user_query)
    intent = detect_intent(user_query)

    module_id = route_to_module(intent, emotion_data["emotion"], user_query)
    module_data = MODULE_REGISTRY[module_id]

    # ---------------- PROMPT ----------------
    messages = [
        {
            "role": "system",
            "content": """
You are an emotionally intelligent assistant for the NeurOm mental wellness app.

STRICT RULES (MUST FOLLOW):
- ONLY recommend modules from this list:
  Breathing, Morning Meditation, Gratitude, Tratak, Power Nap, Journaling, Affirmations, Sherlock holmes, Cognitive Games, Night Music
- DO NOT mention any other activity, feature, or technique outside this list if asked about total activities or modules available
- DO NOT invent or suggest new modules
- DO NOT use general knowledge to suggest features
- DO NOT mention any game or module that is NOT in the above lists
- DO NOT invent or suggest any new games, modules, or activities
- DO NOT suggest games on every response — ONLY when the user explicitly asks about games or cognitive activities

MODULE PURPOSE GUIDE (use this to explain the recommended module naturally):
- Breathing: instant stress and panic relief, lowers heart rate
- Morning Meditation: calms anxiety, resets the mind at the start of day
- Gratitude: heals sadness, shifts focus to positivity
- Tratak: controls anger and frustration through focused stillness
- Power Nap: recovers from burnout and exhaustion
- Journaling: releases loneliness, suppressed feelings, and chaotic thoughts
- Affirmations: builds confidence and replaces negative self-talk
- Sherlock Holmes: breaks overthinking loops through logical engagement
- Night Music: helps with sleep issues and racing thoughts at night

GAME SUGGESTION RULES:
- If user asks "what games are available?" or "suggest a game" or "cognitive games" → list ONLY the 10 games above with a 1-line description
- MindFlip: card matching memory game
- NumberNest: number puzzle and logic challenge
- WordHunt: hidden word vocabulary game
- AlphaQuest: pattern and word uncovering game
- PercentPro: pie chart and percentage decision game
- NumberStorm: dynamic number puzzle race
- BallRush: reflex and spatial awareness runner
- RushHour: three-lane reflex action game
- StackUp: stacking precision and timing game
- BrickBreaker: arcade brick smashing game


CRITICAL RULE — MODULE CONSISTENCY:
The system context below will tell you the exact "Recommended Module" for this user.
You MUST reference ONLY that module in your response.
NEVER suggest a different module than what appears in the context.
This is the most important rule — mismatch between your response and the module card
is a serious error.



- Be supportive, natural, and conversational.
- Give meaningful responses (not short robotic replies).
- You can comfort, guide, and explain briefly.
- DO NOT explain navigation or app paths.
- DO NOT suggest games unless the user specifically asks about them.

KNOWLEDGE QUERY RULES (when knowledge context is provided):
- Use the provided book knowledge to give accurate, helpful answers
- Explain concepts in simple, warm language — not academic tone
- Always relate the answer back to the user's wellbeing
- End with a relevant module suggestion from the allowed list
"""
        }
    ]

    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["message"]
        })

    # ---------------- RAG CONTEXT (only for knowledge queries) ----------------
    rag_context = ""
    if intent == "knowledge_query" and RETRIEVER_INSTANCE is not None:
        try:
            docs = RETRIEVER_INSTANCE.get_relevant_documents(user_query)
            if docs:
                rag_context = "\n\n".join([doc.page_content for doc in docs[:3]])
                print(f"RAG: Retrieved {len(docs[:3])} chunks for knowledge query")
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            rag_context = ""

    # ---------------- BUILD CONTEXT MESSAGE ----------------
    context_content = f"""
Detected Emotion: {emotion_data['emotion']}
Detected Intent: {intent}
Recommended Module: {module_data['module_name']}
"""

    if rag_context:
        context_content += f"""

You have access to the following knowledge from trusted mindfulness and wellness books.
Use this knowledge to answer the user's question in a warm, supportive, conversational tone.
DO NOT copy text directly. Summarize and explain naturally.

--- KNOWLEDGE CONTEXT ---
{rag_context}
--- END OF CONTEXT ---
"""

    messages.append({
        "role": "system",
        "content": context_content
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
        "rag_used": intent == "knowledge_query" and bool(rag_context),
        "primary_recommendation": {
            "module_id": module_id,
            "module_name": module_data["module_name"],
            "category": module_data["category"],
            "action": "open_module"
        }
    }

if not RESOURCES_INITIALIZED:
    initialize_resources()