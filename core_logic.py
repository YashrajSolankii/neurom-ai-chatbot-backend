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
SESSION_LAST_MODULE = {}   # tracks last recommended module per session
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
    "breatheeasy_relax":        {"module_name": "Breathing",          "category": "emotional_regulation"},
    "morning_meditation_guided":{"module_name": "Morning Meditation",  "category": "emotional_regulation"},
    "gratitude_family":         {"module_name": "Gratitude",           "category": "emotional_regulation"},
    "tratak_focus":             {"module_name": "Tratak",              "category": "emotional_regulation"},
    "power_nap_10":             {"module_name": "Power Nap",           "category": "sleep"},
    "journal":                  {"module_name": "Journaling",          "category": "reflection"},
    "affirmation":              {"module_name": "Affirmations",        "category": "emotional_regulation"},
    "sherlock_holmes":          {"module_name": "Sherlock Holmes",     "category": "cognitive"},
    "cognitive_games":          {"module_name": "Cognitive Games",     "category": "cognitive"},
    "night_music":              {"module_name": "Night Music",         "category": "sleep"},
    "other_music":              {"module_name": "Other Music",         "category": "focus_boost"},
    "mindflip":                 {"module_name": "MindFlip",            "category": "cognitive"},
    "number_nest":              {"module_name": "NumberNest",          "category": "cognitive"},
    "wordhunt":                 {"module_name": "WordHunt",            "category": "cognitive"},
    "alphaquest":               {"module_name": "AlphaQuest",          "category": "cognitive"},
    "percentpro":               {"module_name": "PercentPro",          "category": "cognitive"},
    "numberstorm":              {"module_name": "NumberStorm",         "category": "cognitive"},
    "ballrush":                 {"module_name": "BallRush",            "category": "cognitive"},
    "rushhour":                 {"module_name": "RushHour",            "category": "cognitive"},
    "stackup":                  {"module_name": "StackUp",             "category": "cognitive"},
    "brickbreaker":             {"module_name": "BrickBreaker",        "category": "cognitive"},
}

# ---------------- MEMORY HELPERS ----------------
def generate_session_id():
    return str(uuid.uuid4())

def get_session_history(session_id: str):
    return SESSION_MEMORY.get(session_id, [])

def update_session_history(session_id: str, role: str, message: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    SESSION_MEMORY[session_id].append({"role": role, "message": message})
    SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-MEMORY_WINDOW:]

# ---------------- EMOTION DETECTION ----------------
import re

def normalize_text(text: str):
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_emotion(text: str):
    try:
        clean_text = normalize_text(text)
        results = emotion_classifier(clean_text)[0]

        label_mapping = {
            "anger": "anger", "disgust": "anger",
            "fear": "anxiety", "joy": "positive",
            "neutral": "neutral", "sadness": "sadness",
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

        sadness_patterns  = ["feel nothing", "going through", "empty", "numb", "no purpose", "pointless", "lost interest"]
        anxiety_patterns  = ["mind won't stop", "can't stop thinking", "thoughts keep", "over and over", "replaying", "won't slow down"]
        burnout_patterns  = ["always tired", "no energy", "drained", "exhausted", "burnt out"]

        pattern_emotion = None
        pattern_strength = 0

        if any(p in clean_text for p in sadness_patterns):
            pattern_emotion, pattern_strength = "sadness", 0.75
        elif any(p in clean_text for p in anxiety_patterns):
            pattern_emotion, pattern_strength = "anxiety", 0.75
        elif any(p in clean_text for p in burnout_patterns):
            pattern_emotion, pattern_strength = "burnout", 0.75

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
        intensity = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"

        return {"emotion": final_emotion, "confidence": confidence, "intensity": intensity}

    except Exception as e:
        print("Emotion detection error:", e)
        return {"emotion": "neutral", "confidence": 0.5, "intensity": "low"}

# ---------------- INTENT DETECTION ----------------
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
        result = intent_classifier(clean_text, candidate_labels, multi_label=False)
        top_label = result["labels"][0]
        score = result["scores"][0]

        mapping = {
            "user needs help calming down":            "breathing_request",
            "user is feeling anxious or overwhelmed":  "meditation_request",
            "user feels sad or emotionally low":       "sherlock_request",
            "user wants to sleep or relax":            "sleep_request",
            "user feels lonely or isolated":           "journaling_request",
            "user wants motivation or confidence":     "affirmation_request",
            "user is overthinking or stuck in thoughts": "sherlock_request",
            "user wants to improve focus":             "cognitive_training",
            "user wants relaxing music":               "music_request",
            "user is asking for knowledge or explanation": "knowledge_query"
        }

        if score < 0.4:
            return "emotional_regulation"

        return mapping.get(top_label, "emotional_regulation")

    except Exception as e:
        print("Intent detection error:", e)
        return "emotional_regulation"

# ---------------- GREETING DETECTION ----------------
GREETING_PATTERNS = [
    "hi", "hey", "hello", "hii", "heyy", "heyyy", "sup", "what's up",
    "whats up", "yo", "good morning", "good evening", "good afternoon",
    "good night", "howdy", "greetings", "namaste", "hola",
]

SMALL_TALK_PATTERNS = [
    "how are you", "how r u", "how are u", "what are you", "who are you",
    "tell me about yourself", "are you a bot", "are you ai", "are you real",
]

def is_greeting_or_small_talk(text: str) -> bool:
    clean = text.lower().strip().rstrip("!?.").strip()
    if clean in GREETING_PATTERNS:
        return True
    words = clean.split()
    if len(words) <= 3 and any(clean.startswith(g) for g in GREETING_PATTERNS):
        return True
    if any(pattern in clean for pattern in SMALL_TALK_PATTERNS):
        return True
    return False

# ---------------- ENQUIRY DETECTION ----------------
# Messages that need an answer but NO module recommendation card
ENQUIRY_PATTERNS = [
    "what modules", "which modules", "list modules", "what features",
    "what can you do", "what do you offer", "what is available",
    "how many modules", "tell me about modules", "what activities",
    "which module", "what module", "top modules", "best modules",
    "top 5", "best 5", "suggest me", "recommend me a module",
    "what games", "which games", "list games", "available games",
    "how does this work", "how do you work", "what is neurom",
    "why should i use", "why use neurom", "what is this app",
    "how can you help", "explain neurom",
    "which is better", "what should i use", "difference between",
    "what is mindfulness", "what is meditation", "how to meditate",
    "what is breathing", "how does breathing help", "what is journaling",
    "see you later", "i will use you later", "maybe later", "will try later",
    "will use it", "i will think", "let me think", "will come back",
    "not now", "some other time", "another day",
    "goodbye", "bye", "take care",
    "thank you", "thanks a lot", "thanks", "got it", "understood",
    "makes sense", "okay i see", "i see", "noted",
    "yaa great", "yaa sure", "yaa okay", "sounds good",
    "i will try", "i'll try", "will definitely",
]

ACKNOWLEDGEMENT_STARTERS = [
    "ok", "okay", "fine", "sure", "alright", "got it", "noted",
    "thanks", "thank you", "cool", "nice", "great", "awesome",
    "understood", "yep", "yaa", "yeah", "yes", "nope", "no",
    "hmm", "ohh", "oh", "i see", "makes sense", "sounds good",
]

def is_enquiry_or_no_recommendation_needed(text: str) -> bool:
    clean = text.lower().strip()
    if any(pattern in clean for pattern in ENQUIRY_PATTERNS):
        return True
    words = clean.split()
    if len(words) <= 5 and any(clean.startswith(a) for a in ACKNOWLEDGEMENT_STARTERS):
        return True
    return False

# ---------------- EMOTIONAL CONTENT CHECK ----------------
EMOTIONAL_KEYWORDS = [
    "sad", "sadness", "depressed", "crying", "cry", "tears",
    "anxious", "anxiety", "nervous", "scared", "fear", "worried", "worry",
    "stressed", "stress", "overwhelmed", "panic",
    "angry", "anger", "furious", "irritated", "frustrated", "rage",
    "lonely", "alone", "isolated", "empty", "numb",
    "tired", "exhausted", "drained", "burnout", "burnt out",
    "hopeless", "worthless", "useless", "failure", "hate myself",
    "overthinking", "can't stop", "mind won't", "thoughts keep",
    "giving up", "give up", "can't cope", "can't do this",
    "nothing is going", "feel like", "feeling like",
    "dont know", "don't know", "lost", "confused", "stuck",
    "struggling", "sleep", "insomnia", "can't sleep",
    "low", "down", "not okay", "not good", "not well",
    "hurt", "pain", "suffering", "heartbroken", "feels like",
]

def has_emotional_content(text: str) -> bool:
    clean = text.lower()
    return any(keyword in clean for keyword in EMOTIONAL_KEYWORDS)

# ---------------- MODULE ROUTING ----------------
def route_to_module(intent: str, emotion: str, user_query: str, session_id: str = "") -> str:
    text = user_query.lower()

    # Intent-based routing (highest priority)
    intent_map = {
        "breathing_request":  "breatheeasy_relax",
        "meditation_request": "morning_meditation_guided",
        "gratitude_request":  "gratitude_family",
        "tratak_request":     "tratak_focus",
        "sleep_request":      "power_nap_10",
        "journaling_request": "journal",
        "affirmation_request":"affirmation",
        "sherlock_request":   "sherlock_holmes",
        "cognitive_training": "cognitive_games",
        "music_request":      "night_music",
    }
    if intent in intent_map:
        candidate = intent_map[intent]
        return _avoid_repeat(candidate, emotion, session_id)

    # Keyword-based routing
    keyword_routing = [
        (["stressed", "stress", "overwhelmed", "pressure", "panic",
          "tight chest", "suffocated", "overloaded", "too much",
          "can't cope", "breathless", "deadline"], "breatheeasy_relax"),

        (["anxious", "anxiety", "nervous", "scared", "fear", "worried",
          "worrying", "dread", "uneasy", "panic attack", "overthinking",
          "on edge", "restless mind", "tense"], "morning_meditation_guided"),

        (["can't sleep", "insomnia", "sleepless", "awake at night",
          "night thoughts", "racing thoughts at night", "sleep problem",
          "difficulty sleeping", "restless night"], "night_music"),

        (["burnout", "burnt out", "exhausted", "drained", "no energy",
          "mentally tired", "fatigue", "worn out", "sluggish",
          "lethargic", "energy crash"], "power_nap_10"),

        (["lonely", "alone", "isolated", "no one understands",
          "feel invisible", "disconnected", "left out", "abandoned",
          "no one cares", "feel empty inside", "no one to talk to",
          "suppressed", "unheard"], "journal"),

        (["angry", "anger", "furious", "irritated", "frustrated",
          "rage", "mad", "annoyed", "aggressive", "hostile",
          "resentment", "bitter"], "tratak_focus"),

        (["not good enough", "worthless", "hate myself", "confidence",
          "self doubt", "insecure", "i can't do anything", "failure",
          "loser", "useless", "no self worth", "not capable"], "affirmation"),

        (["overthinking", "can't stop thinking", "mind won't stop",
          "thoughts keep", "over and over", "replaying", "mental loop",
          "can't decide", "stuck in my head", "circular thoughts",
          "thinking too much"], "sherlock_holmes"),

        (["sad", "sadness", "depressed", "hopeless", "empty", "numb",
          "no purpose", "pointless", "lost interest", "nothing matters",
          "feel nothing", "meaningless", "joyless", "melancholy",
          "heartbroken", "grief", "feel down", "low mood",
          "giving up", "give up"], "gratitude_family"),

        (["focus", "concentration", "distracted", "brain fog", "study",
          "work music", "productivity", "attention", "procrastinating",
          "can't focus", "addiction", "craving", "lo-fi", "lofi",
          "frequency music"], "other_music"),
    ]

    for keywords, module in keyword_routing:
        if any(w in text for w in keywords):
            return _avoid_repeat(module, emotion, session_id)

    # Emotion-based fallback — NO morning_meditation as default
    emotion_fallback = {
        "anxiety":  "breatheeasy_relax",
        "stress":   "breatheeasy_relax",
        "burnout":  "power_nap_10",
        "sadness":  "gratitude_family",
        "anger":    "tratak_focus",
        "positive": "affirmation",
        "neutral":  "journal",
    }
    candidate = emotion_fallback.get(emotion, "affirmation")
    return _avoid_repeat(candidate, emotion, session_id)

def _avoid_repeat(candidate: str, emotion: str, session_id: str) -> str:
    """Returns candidate unless it was the last module shown — then picks next best."""
    last = SESSION_LAST_MODULE.get(session_id)
    if not last or last != candidate:
        return candidate

    # Rotation pool per emotion
    rotation = {
        "anxiety":  ["morning_meditation_guided", "breatheeasy_relax", "journal", "affirmation"],
        "sadness":  ["gratitude_family", "journal", "affirmation", "sherlock_holmes"],
        "anger":    ["tratak_focus", "breatheeasy_relax", "journal", "morning_meditation_guided"],
        "burnout":  ["power_nap_10", "breatheeasy_relax", "night_music", "journal"],
        "neutral":  ["journal", "affirmation", "sherlock_holmes", "gratitude_family"],
        "positive": ["affirmation", "gratitude_family", "other_music", "morning_meditation_guided"],
    }
    options = rotation.get(emotion, ["affirmation", "journal", "sherlock_holmes", "gratitude_family"])
    for alt in options:
        if alt != last:
            return alt
    return candidate

# ---------------- CRISIS DETECTION ----------------
def detect_crisis(text: str):
    text = text.lower()

    high_risk_keywords = [
        "kill myself", "suicide", "end my life",
        "want to die", "i want to die",
        "i don't want to live", "harm myself",
        "self harm", "cut myself"
    ]
    medium_risk_keywords = [
        "life is meaningless", "i give up", "can't go on",
        "nothing matters", "hopeless", "tired of everything"
    ]

    for phrase in high_risk_keywords:
        if phrase in text:
            return {"risk_level": "high", "matched_keywords": [phrase]}

    if "die" in text and ("feel" in text or "want" in text):
        return {"risk_level": "high", "matched_keywords": ["implicit_suicidal_intent"]}

    for phrase in medium_risk_keywords:
        if phrase in text:
            return {"risk_level": "medium", "matched_keywords": [phrase]}

    return {"risk_level": "none", "matched_keywords": []}

# ---------------- EXPLICIT MODULE DETECTION ----------------
EXPLICIT_MODULE_MAP = {
    "breathing exercise": "breatheeasy_relax",
    "breathing session":  "breatheeasy_relax",
    "breathing":          "breatheeasy_relax",
    "breathe":            "breatheeasy_relax",
    "breath":             "breatheeasy_relax",
    "morning meditation": "morning_meditation_guided",
    "guided meditation":  "morning_meditation_guided",
    "meditation":         "morning_meditation_guided",
    "meditate":           "morning_meditation_guided",
    "gratitude practice": "gratitude_family",
    "gratitude":          "gratitude_family",
    "tratak":             "tratak_focus",
    "candle gazing":      "tratak_focus",
    "focus meditation":   "tratak_focus",
    "power nap":          "power_nap_10",
    "nap":                "power_nap_10",
    "guided journaling":  "journal",
    "journaling":         "journal",
    "journal":            "journal",
    "diary":              "journal",
    "daily affirmations": "affirmation",
    "affirmations":       "affirmation",
    "affirmation":        "affirmation",
    "sherlock holmes":    "sherlock_holmes",
    "sherlock mode":      "sherlock_holmes",
    "sherlock":           "sherlock_holmes",
    "cognitive games":    "cognitive_games",
    "brain game":         "cognitive_games",
    "night music":        "night_music",
    "sleep music":        "night_music",
    "relaxing music":     "night_music",
    "other music":        "other_music",
    "work music":         "other_music",
    "study music":        "other_music",
    "focus music":        "other_music",
    "frequency music":    "other_music",
    "lo-fi":              "other_music",
    "lofi":               "other_music",
}
def is_user_consenting_to_module(text: str, history: list) -> bool:
    """
    Detects if user is saying yes/agreeing to try a module
    that was suggested in the previous assistant message.
    """
    clean = text.lower().strip().rstrip("!?.")

    consent_words = [
        "yes", "yep", "yeah", "yaa", "sure", "okay", "ok",
        "let's try", "lets try", "i'll try", "ill try",
        "let's do it", "lets do it", "sounds good", "go ahead",
        "start it", "open it", "try that", "try it",
        "i want to try", "show me", "let's go", "lets go",
    ]

    # Check if message is a short consent
    is_consent = any(clean == w or clean.startswith(w) for w in consent_words)
    if not is_consent:
        return False

    # Check if last assistant message was recommending a module
    if not history:
        return False

    last_assistant_msgs = [
        m["message"] for m in history
        if m["role"] == "assistant"
    ]
    if not last_assistant_msgs:
        return False

    last_msg = last_assistant_msgs[-1].lower()

    # If assistant mentioned a module name in its last message
    module_names = [
        "breathing", "meditation", "gratitude", "tratak",
        "power nap", "journaling", "affirmation", "sherlock",
        "cognitive", "night music", "other music", "module"
    ]
    return any(name in last_msg for name in module_names)

# ---------------- LLM-BASED RECOMMENDATION GATE ----------------
def should_recommend_module(user_query: str, history: list) -> bool:
    """
    Uses LLM to decide if this message genuinely needs a module recommendation.
    Returns True only if the user is expressing an emotional/wellness need.
    Returns False for enquiries, questions, acknowledgements, app feedback etc.
    """
    try:
        history_text = ""
        if history:
            last_two = history[-2:]
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['message']}" for m in last_two
            ])

        check = LLM_INSTANCE.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a classifier for a mental wellness chatbot.
Your job is to decide if the user's message expresses a genuine emotional or wellness need
that warrants recommending a wellness module (like breathing, meditation, journaling etc).

Answer ONLY with one word: YES or NO

Answer YES if the message:
- Expresses a negative emotion (sadness, anxiety, stress, anger, loneliness, burnout)
- Describes a personal struggle or difficult feeling
- Asks for help with a specific emotional problem
- Mentions sleep issues, overthinking, exhaustion, grief

Answer NO if the message:
- Is a question about the app, features, or modules
- Is asking for information or explanation about anything
- Is an acknowledgement, farewell, or reaction (ok, thanks, bye, yaa great, got it)
- Is feedback about the app (pros, cons, benefits, review)
- Is asking about what you can do or what is available
- Is a casual follow-up that does not express personal distress
- Contains words like: benefits, cons, pros, features, explain, what is, how does, tell me about, list, top, best, compare

Conversation context (last 2 messages):
""" + history_text
                },
                {
                    "role": "user",
                    "content": f"User message: {user_query}\nShould I recommend a wellness module? Answer YES or NO only."
                }
            ],
            max_tokens=5,
            temperature=0.0
        )

        answer = check.choices[0].message.content.strip().upper()
        print(f"Module gate decision for '{user_query[:40]}': {answer}")
        return answer.startswith("YES")

    except Exception as e:
        print(f"Module gate error: {e}")
        # If LLM call fails, fall back to emotional content check
        return has_emotional_content(user_query)

def detect_explicit_module(text: str) -> Optional[str]:
    clean = text.lower().strip()
    for phrase in sorted(EXPLICIT_MODULE_MAP.keys(), key=len, reverse=True):
        if phrase in clean:
            return EXPLICIT_MODULE_MAP[phrase]
    return None

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

    # ── 1. CRISIS CHECK — always first ──
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
            "safe_mode": True,
            "rag_used": False,
            "primary_recommendation": None,
        }

    # ── 2. GREETING / SMALL TALK — no module card ──
    if is_greeting_or_small_talk(user_query):
        update_session_history(session_id, "user", user_query)
        greeting_response = LLM_INSTANCE.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are NeurOm, a warm and empathetic mental wellness companion.
The user has sent a greeting or casual message. Respond warmly and naturally.
Ask ONE gentle open-ended follow-up question to understand how they are feeling today.
Do NOT recommend any modules or activities yet.
Keep response short — 2 to 3 sentences maximum.
IMPORTANT: Look at the conversation history and NEVER repeat a question you already asked."""
                },
                *[{"role": m["role"], "content": m["message"]} for m in history],
                {"role": "user", "content": user_query}
            ],
            temperature=0.6
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
            "primary_recommendation": None,
        }

    # ── 3. ENQUIRY / ACKNOWLEDGEMENT — answer but no module card ──
    if is_enquiry_or_no_recommendation_needed(user_query):
        update_session_history(session_id, "user", user_query)
        enquiry_response = LLM_INSTANCE.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are NeurOm, a warm and empathetic mental wellness companion.
The user has sent an enquiry, acknowledgement, or farewell message.
Answer helpfully and naturally. If it is a farewell, respond warmly and briefly.
Do NOT proactively recommend any module unless the user explicitly asks for one.
Keep the response conversational and natural — not robotic.
STRICT: Do NOT mention any specific module name unless directly asked.
Available modules if asked: Breathing, Morning Meditation, Gratitude, Tratak,
Power Nap, Journaling, Affirmations, Sherlock Holmes, Cognitive Games, Night Music, Other Music."""
                },
                *[{"role": m["role"], "content": m["message"]} for m in history],
                {"role": "user", "content": user_query}
            ],
            temperature=0.5
        )
        answer = enquiry_response.choices[0].message.content
        update_session_history(session_id, "assistant", answer)
        return {
            "session_id": session_id,
            "response": answer,
            "emotion_detected": "neutral",
            "intent": "enquiry",
            "confidence": 1.0,
            "intensity": "low",
            "safe_mode": False,
            "rag_used": False,
            "primary_recommendation": None,
        }

    # ── 4. EMOTION DETECTION ──
    emotion_data = detect_emotion(user_query)

    # ── 5. EXPLICIT MODULE REQUEST ──
    explicit_module_id = detect_explicit_module(user_query)
    if explicit_module_id:
        intent = "explicit_module_request"
        module_id = explicit_module_id
        print(f"Explicit module detected: {module_id}")
    else:
        intent = detect_intent(user_query)

        # ── 6. LLM GATE — decides if module card is needed ──
        # First check if user is consenting to a previously suggested module
        if is_user_consenting_to_module(user_query, history):
            show_recommendation = True
        else:
            show_recommendation = should_recommend_module(user_query, history)

        if not show_recommendation:
            update_session_history(session_id, "user", user_query)
            plain_response = LLM_INSTANCE.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """You are NeurOm, a warm mental wellness companion.
Respond naturally and helpfully to the user's message.
If it is a question, answer it. If it is a farewell, respond warmly.
If it is feedback about the app, acknowledge it gracefully.
Do NOT recommend any specific module in your response.
Keep the response conversational and natural."""
                    },
                    *[{"role": m["role"], "content": m["message"]} for m in history],
                    {"role": "user", "content": user_query}
                ],
                temperature=0.5
            )
            answer = plain_response.choices[0].message.content
            update_session_history(session_id, "assistant", answer)
            return {
                "session_id": session_id,
                "response": answer,
                "emotion_detected": emotion_data["emotion"],
                "intent": intent,
                "confidence": emotion_data["confidence"],
                "intensity": emotion_data["intensity"],
                "safe_mode": False,
                "rag_used": False,
                "primary_recommendation": None,
            }

        module_id = route_to_module(intent, emotion_data["emotion"], user_query, session_id)

    # Track last module per session to avoid repetition
    SESSION_LAST_MODULE[session_id] = module_id
    module_data = MODULE_REGISTRY[module_id]

    # ── 7. BUILD PROMPT ──
    messages = [
        {
            "role": "system",
            "content": """You are an emotionally intelligent assistant for the NeurOm mental wellness app.

STRICT RULES:
- ONLY recommend modules from this list:
  Breathing, Morning Meditation, Gratitude, Tratak, Power Nap, Journaling,
  Affirmations, Sherlock Holmes, Cognitive Games, Night Music, Other Music
- DO NOT invent or suggest new modules
- DO NOT suggest games unless the user explicitly asks about games

MODULE PURPOSE GUIDE:
- Breathing: instant stress and panic relief, lowers heart rate
- Morning Meditation: calms anxiety, resets the mind
- Gratitude: heals sadness, shifts focus to positivity
- Tratak: controls anger and frustration through focused stillness
- Power Nap: recovers from burnout and exhaustion
- Journaling: releases loneliness and suppressed feelings
- Affirmations: builds confidence and replaces negative self-talk
- Sherlock Holmes: breaks overthinking loops through logical engagement
- Night Music: helps with sleep issues and racing thoughts at night
- Other Music: improves focus, aids stress relief through frequency music

CRITICAL RULE — MODULE CONSISTENCY:
The system context below tells you the exact Recommended Module for this user.
You MUST reference ONLY that module in your response.
NEVER suggest a different module than what appears in the context.

- Be supportive, natural, and conversational.
- Give meaningful responses — not short robotic replies.
- DO NOT explain navigation or app paths.

KNOWLEDGE QUERY RULES (when knowledge context is provided):
- Use the provided book knowledge to give accurate, helpful answers
- Explain in simple warm language — not academic tone
- Always relate the answer back to the user's wellbeing
- End with a relevant module suggestion from the allowed list"""
        }
    ]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["message"]})

    # ── 8. RAG CONTEXT ──
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

    # ── 9. CONTEXT MESSAGE ──
    context_content = f"""
Detected Emotion: {emotion_data['emotion']}
Detected Intent: {intent}
Recommended Module: {module_data['module_name']}
IMPORTANT: The only module you are allowed to mention or recommend in your response is "{module_data['module_name']}".
Do NOT mention any other module name.
Your response must naturally guide the user toward "{module_data['module_name']}" only.
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

    messages.append({"role": "system", "content": context_content})
    messages.append({"role": "user", "content": user_query})

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
