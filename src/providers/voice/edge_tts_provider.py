"""Microsoft neural TTS via edge-tts; no API key required."""
import asyncio
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import edge_tts
from src.models.base_provider import ProviderStatus
from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice

def plain(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def chunks(text, limit=2400):
    text = plain(text)
    if not text:
        raise ValueError("Narration text is empty")
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
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

class EdgeTTSVoiceProvider(IVoiceProvider):
    def __init__(self, config):
        self.config = config
    async def validate(self) -> bool:
        return True
    async def get_status(self) -> ProviderStatus:
        return ProviderStatus(True, "edge-tts requires internet access")
    def get_capabilities(self) -> List[str]:
        return ["text_to_speech", "russian", "english"]
    def get_config(self) -> Dict[str, Any]:
        return {"provider_type": "edge_tts"}
    async def list_available_voices(self) -> List[Voice]:
        return [
            Voice(voice_id="ru-RU-SvetlanaNeural", name="Svetlana Neural", language="ru"),
            Voice(voice_id="ru-RU-DmitryNeural", name="Dmitry Neural", language="ru"),
            Voice(voice_id="en-US-JennyNeural", name="Jenny Neural", language="en")
        ]
    async def generate_voice(self, text: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        options = options or VoiceOptions(language="ru")
        voice = options.voice_id or {"ru": "ru-RU-SvetlanaNeural", "en": "en-US-JennyNeural"}.get(options.language, "ru-RU-SvetlanaNeural")
        rate = "%+d%%" % max(-50, min(100, round((options.speed - 1.0) * 100)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_parts = []
            for i, part in enumerate(chunks(text)):
                target = root / ("part_%04d.mp3" % i)
                await edge_tts.Communicate(part, voice, rate=rate).save(str(target))
                if not target.exists() or target.stat().st_size < 100:
                    raise RuntimeError("TTS returned an empty audio chunk")
                audio_parts.append(target)
            if len(audio_parts) == 1:
                audio = audio_parts[0].read_bytes()
            else:
                listing = root / "concat.txt"
                listing.write_text("".join("file '" + p.as_posix() + "'" + chr(10) for p in audio_parts), encoding="utf-8")
                output = root / "narration.mp3"
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c:a", "libmp3lame", "-q:a", "3", str(output),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                _, err = await proc.communicate()
                if proc.returncode != 0 or not output.exists():
                    raise RuntimeError("Failed to join TTS chunks: " + err.decode("utf-8", "ignore")[-500:])
                audio = output.read_bytes()
        return VoiceOutput(audio=audio, format="mp3", metadata={"provider": "edge_tts", "voice_id": voice})
    async def process_ssml(self, ssml: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        return await self.generate_voice(plain(ssml), options)
