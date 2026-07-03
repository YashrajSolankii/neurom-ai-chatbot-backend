# core_logic.py
# ============================================================
# UPGRADES APPLIED (everything else is identical to the original):
#
# UPGRADE 1 — Emotion model:
#   OLD: j-hartmann/emotion-english-distilroberta-base (7 labels, distilled)
#   NEW: SamLowe/roberta-base-go_emotions (28 nuanced labels, full RoBERTa)
#   The 28 labels are mapped back to the same 5 internal categories the rest
#   of the code already uses, so nothing downstream breaks.
#
# UPGRADE 2 — Intent / module classification:
#   OLD: facebook/bart-large-mnli zero-shot (slow, generic NLI model)
#   NEW: Semantic cosine similarity using the sentence-transformers embedding
#        model already loaded for RAG — zero new dependencies, faster, more
#        accurate for short-text-to-category matching.
#   The similarity score supplements the existing keyword routing and LLM gate
#   logic; it does not replace them. All original routing rules are preserved.
#
# LLM BACKEND:
#   OLD: Groq (llama-3.1-8b-instant)
#   NEW: Featherless.ai (meta-llama/Llama-3.3-70B-Instruct) — flat monthly
#        subscription, OpenAI-compatible API, no per-token billing.
#        Uses FEATHERLESS_API_KEY from .env instead of GROQ_API_KEY.
# ============================================================

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import uuid
import re
import numpy as np
from typing import Optional
from transformers import pipeline
import torch

print("core_logic.py: Loading environment variables...")
load_dotenv()

# ---------------- DEVICE SETUP ----------------
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Emotion model running on: {'GPU' if DEVICE == 0 else 'CPU'}")

# ---------------- UPGRADE 1: EMOTION CLASSIFIER ----------------
# Full RoBERTa fine-tuned on Google's GoEmotions dataset.
# 28 nuanced labels → mapped to the same 5 internal categories as before.
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=DEVICE
)

# Maps all 28 GoEmotions labels to the 5 categories the rest of the app uses.
GOEMOTIONS_TO_INTERNAL = {
    "anger":          "anger",
    "annoyance":      "anger",
    "disapproval":    "anger",
    "disgust":        "anger",
    "fear":           "anxiety",
    "nervousness":    "anxiety",
    "embarrassment":  "anxiety",
    "confusion":      "anxiety",
    "joy":            "positive",
    "amusement":      "positive",
    "admiration":     "positive",
    "approval":       "positive",
    "excitement":     "positive",
    "gratitude":      "positive",
    "love":           "positive",
    "optimism":       "positive",
    "pride":          "positive",
    "relief":         "positive",
    "caring":         "positive",
    "desire":         "positive",
    "curiosity":      "positive",
    "realization":    "positive",
    "sadness":        "sadness",
    "grief":          "sadness",
    "disappointment": "sadness",
    "remorse":        "sadness",
    "neutral":        "neutral",
    "surprise":       "neutral",
}

# ---------------- FEATHERLESS CLIENT ----------------
# Flat-rate subscription — no per-token billing.
# Uses the OpenAI SDK pointed at Featherless's OpenAI-compatible endpoint.
LLM_INSTANCE = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.getenv("FEATHERLESS_API_KEY")
)
CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# ---------------- GLOBAL MEMORY STORE ----------------
SESSION_MEMORY = {}
SESSION_LAST_MODULE = {}
MEMORY_WINDOW = 6

RETRIEVER_INSTANCE = None
EMBEDDINGS_INSTANCE = None   # kept globally so classify_module_intent can reuse it
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
    "breatheeasy_relax":         {"module_name": "Breathing",          "category": "emotional_regulation"},
    "morning_meditation_guided": {"module_name": "Morning Meditation", "category": "emotional_regulation"},
    "gratitude_family":          {"module_name": "Gratitude",          "category": "emotional_regulation"},
    "tratak_focus":              {"module_name": "Tratak",             "category": "emotional_regulation"},
    "power_nap_10":               {"module_name": "Power Nap",          "category": "sleep"},
    "journal":                    {"module_name": "Journaling",         "category": "reflection"},
    "affirmation":                {"module_name": "Affirmations",       "category": "emotional_regulation"},
    "sherlock_holmes":            {"module_name": "Sherlock Holmes",    "category": "cognitive"},
    "cognitive_games":            {"module_name": "Cognitive Games",    "category": "cognitive"},
    "night_music":                {"module_name": "Night Music",        "category": "sleep"},
    "other_music":                {"module_name": "Other Music",        "category": "focus_boost"},
    "mindflip":                   {"module_name": "MindFlip",           "category": "cognitive"},
    "number_nest":                {"module_name": "NumberNest",         "category": "cognitive"},
    "wordhunt":                   {"module_name": "WordHunt",           "category": "cognitive"},
    "alphaquest":                 {"module_name": "AlphaQuest",         "category": "cognitive"},
    "percentpro":                 {"module_name": "PercentPro",         "category": "cognitive"},
    "numberstorm":                {"module_name": "NumberStorm",        "category": "cognitive"},
    "ballrush":                   {"module_name": "BallRush",           "category": "cognitive"},
    "rushhour":                   {"module_name": "RushHour",           "category": "cognitive"},
    "stackup":                    {"module_name": "StackUp",            "category": "cognitive"},
    "brickbreaker":                {"module_name": "BrickBreaker",       "category": "cognitive"},
    "selfigo_soundscapes":         {"module_name": "Selfigo Soundscapes",  "category": "grounding"},
    "chakra_music":                {"module_name": "Chakra Music",          "category": "energy_balance"},
}

# ---------------- UPGRADE 2: SEMANTIC SIMILARITY MODULE MATCHING ----------------
# Descriptions used to pre-compute module embeddings at startup.
# The embedding model is the same one already loaded for RAG — zero new deps.
MODULE_DESCRIPTIONS = {
    "breatheeasy_relax":         "instant relief for stress, panic, feeling overwhelmed",
    "morning_meditation_guided": "calming anxiety, resetting an overactive or racing mind",
    "gratitude_family":          "helping with sadness by shifting focus to positives, feeling thankful",
    "tratak_focus":              "a focused gazing practice that helps with anger and frustration",
    "power_nap_10":               "recovering from burnout, exhaustion, fatigue, feeling tired",
    "journal":                    "a private space to process loneliness or suppressed feelings",
    "affirmation":                "building confidence, countering negative self-talk, low self-esteem",
    "sherlock_holmes":            "logic puzzles that interrupt overthinking loops, racing thoughts",
    "cognitive_games":            "light mental exercises to build focus and concentration",
    "night_music":                "curated calming audio for sleep issues and racing thoughts at night",
    "other_music":                "focus-boosting frequency music for concentration and productivity",
    "selfigo_soundscapes":         "disconnected from nature, need grounding, craving stillness, overstimulated senses, feeling stagnant, need inner cleansing, spiritual disconnection, feeling scattered energy, craving natural sounds, seeking inner balance",
    "chakra_music":                "feeling unsafe or insecure, low self-worth, blocked creativity, emotional numbness, lack of confidence, difficulty expressing feelings, foggy intuition, disconnected from purpose, energy feels imbalanced, need whole-body balance",
}

MODULE_EMBEDDINGS = {}   # pre-computed in initialize_resources()


def _cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def classify_module_intent(text: str, top_k: int = 2):
    """Returns top-k modules by cosine similarity to the user's message.
    Used as a supplementary signal alongside keyword routing — the existing
    routing logic still has priority; this adds a numeric score for debugging
    and for cases where keywords don't fire."""
    if EMBEDDINGS_INSTANCE is None or not MODULE_EMBEDDINGS:
        return []
    try:
        query_vec = EMBEDDINGS_INSTANCE.embed_query(text)
        scored = sorted(
            [(mid, _cosine_similarity(query_vec, vec)) for mid, vec in MODULE_EMBEDDINGS.items()],
            key=lambda x: x[1], reverse=True
        )
        return [
            {"module_id": mid, "module_name": MODULE_REGISTRY[mid]["module_name"], "similarity": round(s, 3)}
            for mid, s in scored[:top_k]
        ]
    except Exception as e:
        print("Module similarity error:", e)
        return []

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

# ---------------- TEXT NORMALIZATION ----------------
def normalize_text(text: str):
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- EMOTION DETECTION (Upgrade 1 applied) ----------------
def detect_emotion(text: str):
    try:
        clean_text = normalize_text(text)
        results = emotion_classifier(clean_text)[0]

        best = max(results, key=lambda x: x["score"])
        mapped_emotion = GOEMOTIONS_TO_INTERNAL.get(best["label"], "neutral")
        model_confidence = float(best["score"])

        # ── Original negative-signal override (preserved exactly) ──
        negative_signals = [
            "nothing", "no one", "never", "not working",
            "not going", "wrong", "off", "bad", "stuck"
        ]
        if any(word in clean_text for word in negative_signals):
            if mapped_emotion in ["neutral", "positive"]:
                mapped_emotion = "sadness"
                model_confidence = max(model_confidence, 0.65)

        # ── Original pattern matching (preserved exactly) ──
        sadness_patterns = ["feel nothing", "going through", "empty", "numb", "no purpose", "pointless", "lost interest"]
        anxiety_patterns = ["mind won't stop", "can't stop thinking", "thoughts keep", "over and over", "replaying", "won't slow down"]
        burnout_patterns = ["always tired", "no energy", "drained", "exhausted", "burnt out"]

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

# ---------------- INTENT DETECTION (Upgrade 2 applied) ----------------
# The old BART-MNLI zero-shot classifier is replaced with semantic similarity.
# The function signature and return values are identical to the original so
# all downstream routing code works without any changes.
def detect_intent(text: str):
    clean_text = normalize_text(text)

    # Intent label → internal intent string (same mapping as original)
    INTENT_DESCRIPTIONS = {
        "breathing_request":    "user needs help calming down from stress or panic",
        "meditation_request":   "user is feeling anxious or overwhelmed and needs to calm their mind",
        "sleep_request":        "user wants to sleep or relax or has sleep problems",
        "journaling_request":   "user feels lonely or isolated and needs to express feelings",
        "affirmation_request":  "user wants motivation or confidence or feels worthless",
        "sherlock_request":     "user is overthinking or stuck in repetitive thoughts",
        "cognitive_training":   "user wants to improve focus or train their brain",
        "music_request":        "user wants relaxing or focus music",
        "knowledge_query":      "user is asking for knowledge or explanation about wellness or mindfulness",
        "emotional_regulation": "user is expressing general emotional distress or sadness",
    }

    if EMBEDDINGS_INSTANCE is None or not MODULE_EMBEDDINGS:
        # Fallback if embeddings not loaded yet
        return "emotional_regulation"

    try:
        query_vec = EMBEDDINGS_INSTANCE.embed_query(clean_text)
        scored = []
        for intent_label, desc in INTENT_DESCRIPTIONS.items():
            if intent_label not in _INTENT_EMBEDDINGS:
                _INTENT_EMBEDDINGS[intent_label] = EMBEDDINGS_INSTANCE.embed_query(desc)
            sim = _cosine_similarity(query_vec, _INTENT_EMBEDDINGS[intent_label])
            scored.append((intent_label, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_label, top_score = scored[0]

        if top_score < 0.35:
            return "emotional_regulation"

        print(f"Intent similarity: {top_label} ({top_score:.3f})")
        return top_label

    except Exception as e:
        print("Intent detection error:", e)
        return "emotional_regulation"

# Cache for intent description embeddings (computed lazily on first call)
_INTENT_EMBEDDINGS = {}

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
        return _avoid_repeat(intent_map[intent], emotion, session_id)

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

        (["disconnected from nature", "need to reset", "feeling unclean", "stuck emotional pattern",
          "craving stillness", "need grounding", "overstimulated", "ocean calm", "need to let go",
          "feeling stagnant", "energy feels blocked", "inner cleansing", "spiritual disconnection",
          "feeling unanchored", "emotional detox", "longing for peace", "out of tune",
          "recharge naturally", "inner balance", "scattered energy", "soothing background",
          "natural sounds", "weighed down", "gentle reset", "harmony within",
          "disjointed", "soft immersion", "calm ambience", "raw emotionally",
          "restorative silence"], "selfigo_soundscapes"),

        (["feeling unsafe", "insecure", "low sense of self-worth", "blocked creativity",
          "emotional numbness", "low personal power", "lack of confidence",
          "difficulty expressing", "feeling unheard", "communication blocks",
          "foggy intuition", "lack of clarity", "disconnected from purpose",
          "closed off emotionally", "feeling ungrounded", "low self-trust",
          "guarded heart", "struggling to speak up", "indecisive", "spiritually stuck",
          "lack of inner alignment", "energy feels imbalanced", "low vitality",
          "blocked self-expression", "unclear thinking", "closed-hearted",
          "lack of inner wisdom", "energetic realignment", "disconnected from higher self",
          "whole-body balance"], "chakra_music"),
    ]

    for keywords, module in keyword_routing:
        if any(w in text for w in keywords):
            return _avoid_repeat(module, emotion, session_id)

    # Semantic similarity fallback (Upgrade 2) — only used when keyword
    # routing doesn't fire, as a smarter alternative to the old emotion
    # fallback for unmapped cases.
    sim_matches = classify_module_intent(user_query, top_k=1)
    if sim_matches and sim_matches[0]["similarity"] >= 0.45:
        candidate = sim_matches[0]["module_id"]
        print(f"Semantic similarity routing: {candidate} ({sim_matches[0]['similarity']})")
        return _avoid_repeat(candidate, emotion, session_id)

    # Emotion-based fallback (preserved from original)
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
    last = SESSION_LAST_MODULE.get(session_id)
    if not last or last != candidate:
        return candidate

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
    "selfigo soundscapes": "selfigo_soundscapes",
    "selfigo":             "selfigo_soundscapes",
    "soundscapes":         "selfigo_soundscapes",
    "nature sounds":       "selfigo_soundscapes",
    "chakra music":        "chakra_music",
    "chakra":              "chakra_music",
    "chakra healing":      "chakra_music",
    "energy music":        "chakra_music",
}

def detect_explicit_module(text: str) -> Optional[str]:
    clean = text.lower().strip()
    for phrase in sorted(EXPLICIT_MODULE_MAP.keys(), key=len, reverse=True):
        if phrase in clean:
            return EXPLICIT_MODULE_MAP[phrase]
    return None

# ---------------- CONSENT DETECTION ----------------
def is_user_consenting_to_module(text: str, history: list) -> bool:
    clean = text.lower().strip().rstrip("!?.")
    consent_words = [
        "yes", "yep", "yeah", "yaa", "sure", "okay", "ok",
        "let's try", "lets try", "i'll try", "ill try",
        "let's do it", "lets do it", "sounds good", "go ahead",
        "start it", "open it", "try that", "try it",
        "i want to try", "show me", "let's go", "lets go",
    ]
    is_consent = any(clean == w or clean.startswith(w) for w in consent_words)
    if not is_consent:
        return False
    if not history:
        return False

    last_assistant_msgs = [m["message"] for m in history if m["role"] == "assistant"]
    if not last_assistant_msgs:
        return False

    last_msg = last_assistant_msgs[-1].lower()
    module_names = [
        "breathing", "meditation", "gratitude", "tratak",
        "power nap", "journaling", "affirmation", "sherlock",
        "cognitive", "night music", "other music", "module"
    ]
    return any(name in last_msg for name in module_names)

# ---------------- LLM-BASED RECOMMENDATION GATE ----------------
def should_recommend_module(user_query: str, history: list) -> bool:
    try:
        history_text = ""
        if history:
            last_two = history[-2:]
            history_text = "\n".join([f"{m['role'].upper()}: {m['message']}" for m in last_two])

        check = LLM_INSTANCE.chat.completions.create(
            model=CHAT_MODEL,
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
        )
        answer = check.choices[0].message.content.strip().upper()
        print(f"Module gate decision for '{user_query[:40]}': {answer}")
        return answer.startswith("YES")

    except Exception as e:
        print(f"Module gate error: {e}")
        return has_emotional_content(user_query)

# ---------------- INITIALIZATION ----------------
def initialize_resources():
    global LLM_INSTANCE, RETRIEVER_INSTANCE, EMBEDDINGS_INSTANCE, MODULE_EMBEDDINGS, RESOURCES_INITIALIZED
    if RESOURCES_INITIALIZED:
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDINGS_INSTANCE = embeddings

    # Pre-compute module description embeddings once at startup (Upgrade 2)
    try:
        for module_id, description in MODULE_DESCRIPTIONS.items():
            MODULE_EMBEDDINGS[module_id] = embeddings.embed_query(description)
        print(f"Module classifier: pre-computed embeddings for {len(MODULE_EMBEDDINGS)} modules")
    except Exception as e:
        print(f"Module embedding pre-computation error: {e}")

    # Load or build ChromaDB vector store
    if os.path.isdir(CHROMA_PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print(f"RAG: Loaded existing vector store from '{CHROMA_PERSIST_DIRECTORY}'")
    else:
        print("RAG: Building vector store from PDFs...")
        all_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for pdf_file in PDF_FILES_CONFIG:
            if os.path.exists(pdf_file):
                try:
                    loader = PyPDFLoader(pdf_file)
                    pages = loader.load()
                    chunks = splitter.split_documents(pages)
                    all_docs.extend(chunks)
                    print(f"  Loaded: {pdf_file} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"  Failed to load {pdf_file}: {e}")
        if all_docs:
            vectorstore = Chroma.from_documents(
                documents=all_docs,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIRECTORY
            )
            print(f"RAG: Built and saved vector store ({len(all_docs)} chunks total)")
        else:
            print("RAG: No PDF documents found. Knowledge queries will use model's general knowledge.")
            vectorstore = None

    RETRIEVER_INSTANCE = vectorstore.as_retriever(search_kwargs={"k": 7}) if vectorstore else None
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

    # ── 2. EXPLICIT MODULE — checked BEFORE greeting/enquiry so phrases like
    #    "ok let's try affirmation" or "start journaling" are never intercepted
    #    by the acknowledgement or enquiry filters. ──
    early_explicit = detect_explicit_module(user_query)
    if early_explicit:
        emotion_data = detect_emotion(user_query)
        module_id = early_explicit
        SESSION_LAST_MODULE[session_id] = module_id
        module_data = MODULE_REGISTRY[module_id]
        print(f"Explicit module detected (pre-filter): {module_id}")
        messages = [
            {
                "role": "system",
                "content": f"""You are an emotionally intelligent assistant for the NeurOm mental wellness app.
The user has explicitly requested the {module_data['module_name']} module.
Respond warmly, briefly describe what this module does, and encourage them to start it.
Do NOT ask follow-up questions. Do NOT recommend any other module.
Keep response to 2-3 sentences."""
            },
            *[{"role": m["role"], "content": m["message"]} for m in history],
            {"role": "user", "content": user_query}
        ]
        completion = LLM_INSTANCE.chat.completions.create(
            model=CHAT_MODEL,
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
            "intent": "explicit_module_request",
            "confidence": emotion_data["confidence"],
            "intensity": emotion_data["intensity"],
            "safe_mode": False,
            "rag_used": False,
            "primary_recommendation": {
                "module_id": module_id,
                "module_name": module_data["module_name"],
                "category": module_data["category"],
                "action": "open_module"
            }
        }

    # ── 3. GREETING / SMALL TALK ──
    if is_greeting_or_small_talk(user_query):
        update_session_history(session_id, "user", user_query)
        greeting_response = LLM_INSTANCE.chat.completions.create(
            model=CHAT_MODEL,
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

    # ── 4. ENQUIRY / ACKNOWLEDGEMENT ──
    if is_enquiry_or_no_recommendation_needed(user_query):
        update_session_history(session_id, "user", user_query)
        enquiry_response = LLM_INSTANCE.chat.completions.create(
            model=CHAT_MODEL,
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

    # ── 5. EMOTION DETECTION ──
    emotion_data = detect_emotion(user_query)

    # ── 5.5. SECONDARY EXPLICIT CHECK (consent patterns like 'yes let's go') ──
    early_explicit = detect_explicit_module(user_query)
    if early_explicit:
        intent = "explicit_module_request"
        module_id = early_explicit
        SESSION_LAST_MODULE[session_id] = module_id
        module_data = MODULE_REGISTRY[module_id]
        print(f"Early explicit module detected: {module_id}")
        messages = [
            {
                "role": "system",
                "content": f"""You are an emotionally intelligent assistant for the NeurOm mental wellness app.
The user has explicitly requested the {module_data['module_name']} module.
Respond warmly, briefly describe what this module does, and encourage them to start it.
Do NOT ask follow-up questions. Do NOT recommend any other module.
Keep response to 2-3 sentences."""
            },
            *[{"role": m["role"], "content": m["message"]} for m in history],
            {"role": "user", "content": user_query}
        ]
        completion = LLM_INSTANCE.chat.completions.create(
            model=CHAT_MODEL,
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
            "intent": "explicit_module_request",
            "confidence": emotion_data["confidence"],
            "intensity": emotion_data["intensity"],
            "safe_mode": False,
            "rag_used": False,
            "primary_recommendation": {
                "module_id": module_id,
                "module_name": module_data["module_name"],
                "category": module_data["category"],
                "action": "open_module"
            }
        }

    # ── 6. EXPLICIT MODULE REQUEST (second check, consent flow) ──
    explicit_module_id = detect_explicit_module(user_query)
    if explicit_module_id:
        intent = "explicit_module_request"
        module_id = explicit_module_id
        print(f"Explicit module detected: {module_id}")
    else:
        intent = detect_intent(user_query)

        # ── 7. LLM GATE ──
        if is_user_consenting_to_module(user_query, history):
            show_recommendation = True
        elif has_emotional_content(user_query) and emotion_data["confidence"] >= 0.55:
            show_recommendation = True
        else:
            show_recommendation = should_recommend_module(user_query, history)

        if not show_recommendation:
            update_session_history(session_id, "user", user_query)
            plain_response = LLM_INSTANCE.chat.completions.create(
                model=CHAT_MODEL,
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

    SESSION_LAST_MODULE[session_id] = module_id
    module_data = MODULE_REGISTRY[module_id]

    # ── 8. BUILD PROMPT ──
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
- Selfigo Soundscapes: nature-based audio for grounding, energetic reset, inner cleansing, and stillness
- Chakra Music: healing frequency music for chakra balance, blocked emotions, and whole-body alignment

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

    # ── 9. RAG CONTEXT ──
    rag_context = ""
    if intent == "knowledge_query" and RETRIEVER_INSTANCE is not None:
        try:
            docs = RETRIEVER_INSTANCE.get_relevant_documents(user_query)
            if docs:
                rag_context = "\n\n".join([doc.page_content for doc in docs[:3]])
                print(f"RAG: Retrieved {len(docs[:3])} chunks for knowledge query")
        except Exception as e:
            print(f"RAG retrieval error: {e}")

    # ── 10. CONTEXT MESSAGE ──
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
        model=CHAT_MODEL,
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