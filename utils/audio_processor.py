import yt_dlp
from pydub import AudioSegment
import os
import uuid
import tempfile

DOWNLOAD_DIR = "downloads"
CHUNK_DIR = "chunks"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)


def _get_cookies_file() -> str | None:
    """Load YouTube cookies from Streamlit secrets if available."""
    try:
        import streamlit as st
        if "youtube" in st.secrets and "cookies" in st.secrets["youtube"]:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            tmp.write(st.secrets["youtube"]["cookies"])
            tmp.close()
            return tmp.name
    except Exception:
        pass

    # Fallback: local cookies.txt file (for local development)
    if os.path.exists("cookies.txt"):
        return "cookies.txt"

    return None


def _download_yt_audio(url: str) -> str:

    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    cookies_file = _get_cookies_file()

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
        "extractor_args": {"youtube": {"player_client": ["web"]}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    # Add cookies if available
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:                         # type: ignore
            info = ydl.extract_info(url, download=True)

            if info is None:
                raise Exception("Failed to extract video info")

            video_id = info.get("id")
            mp3_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.mp3")

            if not os.path.exists(mp3_path):
                raise Exception("MP3 file was not created")

            return mp3_path

    except Exception as e:
        raise Exception(f"YouTube Download Failed: {str(e)}")


def _convert_to_mono_wav(input_path: str) -> str:

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    # Whisper recommended format
    audio = audio.set_frame_rate(16000).set_channels(1)

    audio.export(output_path, format="wav")

    return output_path


def _chunk_audio(wav_path: str, language: str = "en") -> list:

    audio = AudioSegment.from_file(wav_path)

    # Hindi → Sarvam limit
    if language == "hi":
        chunk_ms = 25 * 1000
    else:
        # English → Whisper
        chunk_ms = 10 * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start : start + chunk_ms]

        chunk_path = os.path.join(
            CHUNK_DIR, f"{uuid.uuid4().hex}_chunk_{i+1}.wav"
        )

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str, language: str = "en") -> list:

    # YouTube URL
    if source.startswith("http://") or source.startswith("https://"):

        print("YouTube URL detected. Downloading audio...")

        audio_path = _download_yt_audio(source)

        print("Converting audio to WAV...")

        wav_path = _convert_to_mono_wav(audio_path)

    # Local file
    else:

        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        print("Local file detected. Converting to WAV...")

        wav_path = _convert_to_mono_wav(source)

    print("Chunking audio...")

    chunks = _chunk_audio(wav_path, language)

    print(f"Audio ready. {len(chunks)} chunks created.")

    return chunks