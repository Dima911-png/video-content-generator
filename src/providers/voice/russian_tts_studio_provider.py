"""Remote Russian TTS Studio provider using the VoxCPM2 engine.

The Russian TTS Studio server must be running somewhere reachable from the
video-generation runner. Set RUSSIAN_TTS_STUDIO_URL to its base URL.
"""
import asyncio
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

import requests

from src.models.base_provider import ProviderStatus
from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice


def _plain(text: str) -> str:
    text = re.sub(r"<break\b[^>]*/>", ". ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[(?:Citation|Source)\]", "", text, flags=re.I)
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunks(text: str, limit: int = 1800) -> List[str]:
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


class RussianTTSStudioVoiceProvider(IVoiceProvider):
    """VoxCPM2 voice generation through a hosted Russian TTS Studio instance."""

    def __init__(self, config):
        self.config = config
        self.base_url = (config.api_url or "").strip().rstrip("/")
        self.timeout = max(int(config.timeout or 600), 60)
        self.engine = (config.model_name or "voxcpm").strip().lower()
        self.instruct = (config.extra_params or {}).get(
            "instruct",
            "Russian educational YouTube narration. Natural conversational delivery, "
            "warm confident adult voice, expressive but restrained intonation, clear "
            "pronunciation, medium pace, natural pauses. Do not sound like a voice assistant."
        )

    async def validate(self) -> bool:
        return bool(self.base_url.startswith(("http://", "https://")))

    async def get_status(self) -> ProviderStatus:
        if not await self.validate():
            return ProviderStatus(False, "RUSSIAN_TTS_STUDIO_URL is missing or invalid")
        try:
            response = await asyncio.to_thread(
                requests.get, urljoin(self.base_url + "/", "api/status"), timeout=15
            )
            response.raise_for_status()
            return ProviderStatus(True, "Russian TTS Studio server is reachable")
        except Exception as exc:
            return ProviderStatus(False, f"Russian TTS Studio is unreachable: {exc}")

    def get_capabilities(self) -> List[str]:
        return ["text_to_speech", "russian", "voxcpm2", "long_text_chunking", "natural_prosody"]

    def get_config(self) -> Dict[str, Any]:
        return {"provider_type": "russian_tts_studio", "engine": self.engine, "timeout": self.timeout}

    async def list_available_voices(self) -> List[Voice]:
        return [Voice(voice_id="voxcpm", name="VoxCPM2 — natural Russian narration", language="ru")]

    def _synthesize_chunk(self, text: str, folder: Path, index: int) -> Path:
        endpoint = urljoin(self.base_url + "/", "api/synthesize")
        form = {
            "text": text,
            "engine": self.engine,
            "instruct": self.instruct,
            "speed": "0.95",
            "enable_fallback": "false",
            "enable_postprocess": "true",
            "enable_quality_check": "false",
            "enable_prosody": "true",
            "pause_ms_comma": "140",
            "pause_ms_semicolon": "200",
            "pause_ms_colon": "170",
            "pause_ms_period": "300",
            "pause_ms_exclamation": "320",
            "pause_ms_question": "320",
            "pause_ms_ellipsis": "450",
        }
        try:
            response = requests.post(endpoint, data=form, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Russian TTS Studio synthesis request failed: {exc}") from exc

        audio_url = data.get("audio_url")
        if not audio_url:
            raise RuntimeError("Russian TTS Studio response did not include audio_url.")
        try:
            audio_response = requests.get(urljoin(self.base_url + "/", audio_url.lstrip("/")), timeout=self.timeout)
            # Preserve absolute URLs returned by the server.
            if audio_url.startswith(("http://", "https://")):
                audio_response = requests.get(audio_url, timeout=self.timeout)
            audio_response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Could not download VoxCPM2 audio: {exc}") from exc

        wav_path = folder / f"part_{index:04d}.wav"
        wav_path.write_bytes(audio_response.content)
        if wav_path.stat().st_size < 1000:
            raise RuntimeError("Russian TTS Studio returned an empty audio file.")
        mp3_path = folder / f"part_{index:04d}.mp3"
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
             "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
            capture_output=True, text=True, timeout=self.timeout,
        )
        if proc.returncode != 0 or not mp3_path.exists() or mp3_path.stat().st_size < 1000:
            raise RuntimeError("Could not convert VoxCPM2 audio to MP3: " + proc.stderr[-600:])
        return mp3_path

    async def generate_voice(self, text: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        parts = _chunks(text)
        with tempfile.TemporaryDirectory(prefix="russian_tts_studio_") as temp:
            root = Path(temp)
            files = []
            for i, part in enumerate(parts):
                path = await asyncio.to_thread(self._synthesize_chunk, part, root, i)
                files.append(path)
            if len(files) == 1:
                merged = files[0].read_bytes()
            else:
                listing = root / "concat.txt"
                listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in files), encoding="utf-8")
                output = root / "narration.mp3"
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c:a", "libmp3lame", "-q:a", "2", str(output),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, err = await proc.communicate()
                if proc.returncode != 0 or not output.exists():
                    raise RuntimeError("Could not join VoxCPM2 audio chunks: " + err.decode("utf-8", "ignore")[-800:])
                merged = output.read_bytes()
        return VoiceOutput(audio=merged, format="mp3", metadata={
            "provider": "russian_tts_studio", "engine": self.engine, "chunks": len(parts)
        })

    async def process_ssml(self, ssml: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        return await self.generate_voice(_plain(ssml), options)
