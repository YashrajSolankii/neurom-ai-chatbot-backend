import whisper
from gtts import gTTS
import uuid
import os

print("Loading Whisper model...")

# Load Whisper model once
whisper_model = whisper.load_model("base")

print("Whisper model loaded.")


# Folder to store audio responses
AUDIO_FOLDER = "audio_responses"

if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)


def speech_to_text(audio_path):
    """
    Convert user voice to text
    """
    result = whisper_model.transcribe(audio_path)
    return result["text"]


def text_to_speech(text):
    """
    Convert AI response text to speech
    """

    filename = f"response_{uuid.uuid4()}.mp3"

    filepath = os.path.join(AUDIO_FOLDER, filename)

    tts = gTTS(text=text, lang="en")

    tts.save(filepath)

    return filepath