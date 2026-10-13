"""OpenAI neural text-to-speech provider."""
import asyncio
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

from src.models.base_provider import ProviderStatus
from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice

def _plain(text: str) -> str:
    text = re.sub(r"<break\b[^>]*/>", ". ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _chunks(text: str, limit: int = 3500) -> List[str]:
    text = _plain(text)
    if not text:
        raise ValueError("Narration text is empty.")
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        while len(sentence) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        parts.append(current)
    return parts

class OpenAITTSVoiceProvider(IVoiceProvider):
    """OpenAI speech API; reuses OPENAI_API_KEY used for script generation."""
    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.api_url = (config.api_url or "https://api.openai.com/v1").rstrip("/")
        self.model = config.model_name or "gpt-4o-mini-tts"
        self.timeout = config.timeout
        self.default_voice = (config.extra_params or {}).get("voice", "marin")
        self.instructions = (config.extra_params or {}).get(
            "instructions",
            "Speak in natural, fluent Russian like a professional educational YouTube narrator. "
            "Use clear pronunciation, warm confident delivery, natural intonation and moderate pacing. "
            "Avoid robotic monotony, exaggerated acting, and unnatural pauses. Read numbers and abbreviations naturally."
        )

    async def validate(self) -> bool:
        return bool(self.api_key)

    async def get_status(self) -> ProviderStatus:
        return ProviderStatus(bool(self.api_key), "OpenAI TTS configured" if self.api_key else "OPENAI_API_KEY is missing")

    def get_capabilities(self) -> List[str]:
        return ["text_to_speech", "neural_voice", "multilingual", "long_text_chunking", "style_instructions"]

    def get_config(self) -> Dict[str, Any]:
        return {"provider_type": "openai_tts", "model": self.model, "voice": self.default_voice}

    async def list_available_voices(self) -> List[Voice]:
        return [
            Voice(voice_id="marin", name="Marin — natural narration", language="ru"),
            Voice(voice_id="cedar", name="Cedar — natural narration", language="ru"),
            Voice(voice_id="coral", name="Coral", language="ru"),
            Voice(voice_id="nova", name="Nova", language="ru"),
        ]

    def _request_audio(self, text: str, voice: str, speed: float) -> bytes:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to GitHub Actions secrets.")
        payload = {
            "model": self.model,
            "voice": voice,
            "input": text,
            "instructions": self.instructions,
            "response_format": "mp3",
            "speed": max(0.75, min(1.25, float(speed))),
        }
        try:
            response = requests.post(
                f"{self.api_url}/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAI TTS network error: {exc}") from exc
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get("message", response.text[:800])
            except Exception:
                detail = response.text[:800]
            raise RuntimeError(f"OpenAI TTS returned HTTP {response.status_code}: {detail}")
        if len(response.content) < 1000 or not response.content.startswith(b"ID3") and not response.content.startswith(b"\xff"):
            raise RuntimeError("OpenAI TTS returned an empty or invalid MP3 file.")
        return response.content

    async def generate_voice(self, text: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        options = options or VoiceOptions(language="ru")
        voice = options.voice_id or self.default_voice
        parts = _chunks(text)
        with tempfile.TemporaryDirectory(prefix="openai_tts_") as temp:
            root = Path(temp)
            files = []
            for i, part in enumerate(parts):
                audio = await asyncio.to_thread(self._request_audio, part, voice, options.speed)
                path = root / f"part_{i:04d}.mp3"
                path.write_bytes(audio)
                files.append(path)
            if len(files) == 1:
                merged_bytes = files[0].read_bytes()
            else:
                listing = root / "concat.txt"
                listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in files), encoding="utf-8")
                output = root / "narration.mp3"
                import subprocess
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:a", "libmp3lame", "-q:a", "3", str(output),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, err = await proc.communicate()
                if proc.returncode != 0 or not output.exists():
                    raise RuntimeError("Could not join narration audio chunks: " + err.decode("utf-8", "ignore")[-800:])
                merged_bytes = output.read_bytes()
        return VoiceOutput(audio=merged_bytes, format="mp3", metadata={
            "provider": "openai_tts", "model": self.model, "voice_id": voice, "chunks": len(parts)
        })

    async def process_ssml(self, ssml: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        return await self.generate_voice(_plain(ssml), options)
