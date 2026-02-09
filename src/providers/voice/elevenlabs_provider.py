"""ElevenLabs voice generation provider.

This module provides text-to-speech functionality using the ElevenLabs API,
supporting multiple voices, languages, and voice customization options.

Long scripts are automatically chunked into smaller segments to avoid API
timeouts, then the audio is concatenated.
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List

import requests

from src.models.base_provider import BaseProvider, ProviderStatus
from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice
from src.models.schemas import ProviderConfig

logger = logging.getLogger(__name__)

# Default voice ID (Rachel)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Maximum characters per API call (~500 words ~= 2500 chars).
# ElevenLabs can handle up to 5000 chars but shorter chunks are more reliable.
MAX_CHUNK_CHARS = 2500


def strip_ssml_tags(text: str) -> str:
    """
    Strip SSML tags from text for providers that don't support them.
    
    ElevenLabs has limited SSML support, so we convert to plain text
    while preserving natural pauses via punctuation.
    
    Args:
        text: Text potentially containing SSML markup
        
    Returns:
        Plain text with SSML tags removed
    """
    # Replace break tags with ellipsis for natural pauses
    text = re.sub(r'<break\s+time=["\'][^"\']+["\']\s*/>', '... ', text)
    
    # Remove all other SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove citation markers (various formats)
    text = re.sub(r'\[Citation\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\]', '', text)  # [1], [2], etc.
    text = re.sub(r'\[Source\]', '', text, flags=re.IGNORECASE)
    
    # Clean up any double spaces from removals
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class ElevenLabsVoiceProvider(IVoiceProvider):
    """Voice generation provider using ElevenLabs API."""

    def __init__(self, config: ProviderConfig):
        """
        Initialize ElevenLabs voice provider.

        Args:
            config: Provider configuration with API key and settings.
            
        Note on models:
            - "eleven_multilingual_v2" (default): High quality, works on free tier
            - "eleven_flash_v2_5": Faster, cheaper (~0.25 credit/character)
            - "eleven_monolingual_v1": DEPRECATED - removed from free tier
            
        To use Flash model, set model_name="eleven_flash_v2_5" in config.
        """
        self.config = config
        self.api_key = config.api_key
        self.api_url = config.api_url or "https://api.elevenlabs.io/v1"
        self.model = config.model_name or "eleven_multilingual_v2"
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.extra_params = config.extra_params or {}
        self._voices_cache: Optional[List[Voice]] = None

    async def validate(self) -> bool:
        """
        Validate ElevenLabs configuration.
        
        Note: We don't make an API call to validate since many API keys have
        limited permissions (e.g., only text-to-speech, no voices_read).
        We just check if the key exists and has a valid format.
        """
        if not self.api_key:
            return False
        
        # ElevenLabs API keys start with "sk_" and are typically 32-64 chars
        if not self.api_key.startswith("sk_") or len(self.api_key) < 30:
            return False
            
        return True

    async def get_status(self) -> ProviderStatus:
        """Get current status of ElevenLabs provider."""
        is_valid = await self.validate()
        message = "Connected and ready" if is_valid else "Connection failed or invalid config"
        return ProviderStatus(is_valid, message)

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities."""
        return [
            "text_to_speech",
            "ssml_support",
            "multiple_voices",
            "multiple_languages",
            "voice_customization",
        ]

    def get_config(self) -> Dict[str, Any]:
        """Get configuration without sensitive keys."""
        return {
            "provider_type": self.config.provider_type,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    async def generate_voice(
        self, text: str, options: Optional[VoiceOptions] = None
    ) -> VoiceOutput:
        """Generate voice/audio from text."""
        if not options:
            options = VoiceOptions()

        voice_id = options.voice_id or DEFAULT_VOICE_ID

        for attempt in range(self.max_retries):
            try:
                audio_data = await self._call_elevenlabs(text, voice_id, options)
                return VoiceOutput(
                    audio=audio_data,
                    format=options.extra_options.get("format", "mp3"),
                    metadata={
                        "voice_id": voice_id,
                        "model": self.model,
                        "speed": options.speed,
                        "pitch": options.pitch,
                        "attempt": attempt + 1,
                    },
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def list_available_voices(self) -> List[Voice]:
        """List all available voices."""
        if self._voices_cache:
            return self._voices_cache

        headers = {"xi-api-key": self.api_key}
        response = requests.get(
            f"{self.api_url}/voices",
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        voices = [
            Voice(
                voice_id=v["voice_id"],
                name=v["name"],
                language="en",  # ElevenLabs typically uses English, but could be extended
                metadata={"preview_url": v.get("preview_url", "")},
            )
            for v in data.get("voices", [])
        ]

        self._voices_cache = voices
        return voices

    async def process_ssml(
        self, ssml: str, options: Optional[VoiceOptions] = None
    ) -> VoiceOutput:
        """
        Process SSML markup and generate voice.
        
        Note: ElevenLabs has limited SSML support, so we strip most tags
        and convert to plain text while preserving natural pauses.
        """
        if not options:
            options = VoiceOptions()

        voice_id = options.voice_id or DEFAULT_VOICE_ID
        
        # Strip SSML tags since ElevenLabs has limited support
        plain_text = strip_ssml_tags(ssml)

        for attempt in range(self.max_retries):
            try:
                # Use regular text-to-speech with stripped text
                audio_data = await self._call_elevenlabs(plain_text, voice_id, options)
                return VoiceOutput(
                    audio=audio_data,
                    format=options.extra_options.get("format", "mp3"),
                    metadata={
                        "voice_id": voice_id,
                        "model": self.model,
                        "has_ssml": False,  # Converted to plain text
                        "original_had_ssml": True,
                        "attempt": attempt + 1,
                    },
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _call_elevenlabs(
        self, text: str, voice_id: str, options: VoiceOptions
    ) -> bytes:
        """
        Call ElevenLabs text-to-speech API.
        
        Long texts are automatically split into chunks and the resulting
        audio segments are concatenated so the API never times out.
        """
        chunks = self._split_text(text)
        
        if len(chunks) == 1:
            return self._tts_request(chunks[0], voice_id)
        
        logger.info(f"Splitting {len(text)} chars into {len(chunks)} chunks for TTS")
        audio_parts: List[bytes] = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  Generating audio chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            part = self._tts_request(chunk, voice_id)
            audio_parts.append(part)
        
        return b"".join(audio_parts)

    def _tts_request(self, text: str, voice_id: str) -> bytes:
        """Make a single TTS API request for one chunk of text."""
        headers = {"xi-api-key": self.api_key}

        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.extra_params.get("stability", 0.5),
                "similarity_boost": self.extra_params.get("similarity_boost", 0.75),
            },
        }

        response = requests.post(
            f"{self.api_url}/text-to-speech/{voice_id}",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.content

    @staticmethod
    def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
        """
        Split long text into chunks that respect sentence boundaries.
        
        Tries to split on paragraph breaks first, then sentence endings,
        so the voice output sounds natural at chunk boundaries.
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks: List[str] = []
        remaining = text.strip()
        
        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break
            
            # Try to find a good split point within the limit
            split_at = max_chars
            
            # Prefer paragraph break (\n\n)
            para_break = remaining.rfind("\n\n", 0, max_chars)
            if para_break > max_chars // 3:
                split_at = para_break + 2  # include the newlines
            else:
                # Fall back to sentence ending (. ! ?)
                sentence_end = -1
                for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                    pos = remaining.rfind(punct, 0, max_chars)
                    if pos > sentence_end:
                        sentence_end = pos
                
                if sentence_end > max_chars // 3:
                    split_at = sentence_end + 2  # include punct + space
                else:
                    # Last resort: split on any space
                    space_pos = remaining.rfind(" ", 0, max_chars)
                    if space_pos > 0:
                        split_at = space_pos + 1
            
            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].strip()
        
        return chunks
