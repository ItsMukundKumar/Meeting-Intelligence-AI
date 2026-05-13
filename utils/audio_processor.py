import yt_dlp
from pydub import AudioSegment
from typing import Any
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _download_yt_audio(url: str) -> str:

    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "http_headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True)
        mp3_path = os.path.join(DOWNLOAD_DIR, f"{info['id']}.mp3")
        return mp3_path


def _convert_to_mono_wav(input_path: str) -> str:

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")

    return output_path


def _chunk_audio(wav_path: str, language: str = "en") -> list:
    """
    Chunk audio based on language:
    - English (Whisper) → 10 minute chunks
    - Hindi/Hinglish (Sarvam) → 25 second chunks (Sarvam limit is 30s)
    """
    audio = AudioSegment.from_file(wav_path)

    if language == "hi":
        chunk_ms = 25 * 1000  # 25 seconds for Sarvam
        output_dir = "chunks_sarvam"
    else:
        chunk_ms = 10 * 60 * 1000  # 10 minutes for Whisper
        output_dir = "chunks_whisper"

    os.makedirs(output_dir, exist_ok=True)
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = os.path.join(output_dir, f"chunk_{i+1}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str, language: str = "en") -> list:
    """
    Process a YouTube URL or local file into audio chunks.

    Args:
        source:   YouTube URL or local file path
        language: "en" for English (Whisper), "hi" for Hindi/Hinglish (Sarvam)
    """
    if source.startswith("https://") or source.startswith("http://"):
        print("Youtube URL Detected. Downloading audio.....")
        data = _download_yt_audio(source)
        wav_path = _convert_to_mono_wav(data)
    else:
        print("Local file Detected. Converting to WAV....")
        wav_path = _convert_to_mono_wav(source)

    print("Chunking Audio...")
    chunks = _chunk_audio(wav_path=wav_path, language=language)
    print(f"Audio is Ready - {len(chunks)} chunks created.")
    return chunks
