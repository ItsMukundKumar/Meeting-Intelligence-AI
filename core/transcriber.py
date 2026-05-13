import httpx
import os
from dotenv import load_dotenv
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

load_dotenv()

# ── Config ──────────────────────────────────────────


USE_GROQ = st.secrets.get("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
WHISPER_MODEL = st.secrets.get("WHISPER_MODEL", "base")
SARVAM_API_KEY = st.secrets.get("SARVAM_API_KEY")
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

_model = None
_groq_client = None


# ── Faster Whisper (local) ──────────────────────────

def load_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"Loading Faster Whisper Model: {WHISPER_MODEL}...")
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("Faster Whisper model loaded successfully")
    return _model


def _transcribe_chunk_local(chunk_path: str, language: str = "en", translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    # Pass language hint to Whisper for better accuracy
    lang_code = language if language != "hi" else "hi"
    segments, _ = model.transcribe(
        chunk_path,
        task=task,
        language=lang_code,
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(segment.text for segment in segments).strip()


# ── Groq Whisper (cloud) ────────────────────────────

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _transcribe_chunk_groq(chunk_path: str, language: str = "en") -> str:
    """
    FIX: was hardcoded to language='en'.
    Now passes the actual language so Groq transcribes Hindi correctly too.
    """
    client = get_groq_client()
    # Groq Whisper uses ISO 639-1 codes: 'en', 'hi', etc.
    groq_lang = language if language in ("en", "hi") else "en"
    with open(chunk_path, "rb") as f:
        result = client.audio.transcriptions.create(  # type: ignore[attr-defined]
            file=f,
            model="whisper-large-v3-turbo",
            language=groq_lang,
            response_format="text",
        )
    return result  # type: ignore


# ── Unified chunk transcriber ───────────────────────

def transcribe_chunk(chunk_path: str, language: str = "en", translate: bool = False) -> str:
    if USE_GROQ:
        return _transcribe_chunk_groq(chunk_path, language=language)
    return _transcribe_chunk_local(chunk_path, language=language, translate=translate)


def transcribe_all_chunks(chunks: list, language: str = "en", translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        source = "Groq" if USE_GROQ else "Faster Whisper"
        print(f"Transcribing chunk {i+1}/{len(chunks)} via {source}")
        full_transcript += transcribe_chunk(chunk, language=language, translate=translate) + " "
    print("Transcription Completed")
    return full_transcript.strip()


# ── Sarvam AI (Hindi / Hinglish) ────────────────────

def transcribe_chunk_hindi(chunk_path: str) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in environment variables")

    with open(chunk_path, "rb") as f:
        audio_data = f.read()

    response = httpx.post(
        SARVAM_API_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": (os.path.basename(chunk_path), audio_data, "audio/wav")},
        data={
            "language_code": "hi-IN",
            "model": "saarika:v2.5",
            "with_timestamps": "false",
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("transcript", "")


def translate_hindi_to_english(text: str) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in environment variables")

    MAX_CHARS = 1000
    if len(text) <= MAX_CHARS:
        return _translate_chunk(text)

    sentences = text.replace("।", "।|").split("|")
    translated, batch = "", ""

    for sentence in sentences:
        if len(batch) + len(sentence) <= MAX_CHARS:
            batch += sentence
        else:
            translated += _translate_chunk(batch) + " "
            batch = sentence

    if batch:
        translated += _translate_chunk(batch)

    return translated.strip()


def _translate_chunk(text: str) -> str:
    api_key = SARVAM_API_KEY or ""
    response = httpx.post(
        SARVAM_TRANSLATE_URL,
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        json={
            "input": text,
            "source_language_code": "hi-IN",
            "target_language_code": "en-IN",
            "model": "mayura:v1",
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("translated_text", text)


def transcribe_all_chunks_hindi(chunks: list) -> str:
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY is not set in environment variables")

    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)} via Sarvam AI")
        full_transcript += transcribe_chunk_hindi(chunk) + " "

    print("Sarvam Transcription Completed. Translating to English...")
    english_transcript = translate_hindi_to_english(full_transcript.strip())
    print("Translation Completed.")
    return english_transcript


# ── Unified Entry Point ─────────────────────────────

def transcribe(chunks: list, language: str = "en", translate: bool = False) -> str:
    """
    FIX: Hindi now correctly routes to Sarvam.
    English with USE_GROQ=true now passes language to Groq instead of hardcoding 'en'.
    """
    if language == "hi":
        return transcribe_all_chunks_hindi(chunks)
    return transcribe_all_chunks(chunks, language=language, translate=translate)