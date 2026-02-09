from abc import abstractmethod
from typing import Optional
from .base_provider import BaseProvider


class Voice:
    """Represents an available voice."""

    def __init__(self, voice_id: str, name: str, language: str = "en", **metadata):
        self.voice_id = voice_id
        self.name = name
        self.language = language
        self.metadata = metadata


class VoiceOptions:
    """Options for voice generation."""

    def __init__(
        self,
        voice_id: str = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        language: str = "en",
        **kwargs
    ):
        self.voice_id = voice_id
        self.speed = speed  # 0.5 to 2.0 typically
        self.pitch = pitch  # 0.5 to 2.0 typically
        self.language = language
        self.extra_options = kwargs


class VoiceOutput:
    """Output from voice generation."""

    def __init__(self, audio: bytes, format: str = "mp3", metadata: Optional[dict] = None):
        self.audio = audio
        self.format = format  # 'mp3', 'wav', etc.
        self.metadata = metadata or {}


class IVoiceProvider(BaseProvider):
    """Interface for voice generation providers."""

    @abstractmethod
    async def generate_voice(
        self, text: str, options: Optional[VoiceOptions] = None
    ) -> VoiceOutput:
        """
        Generate voice/audio from text.

        Args:
            text: The text to convert to speech (can include SSML tags).
            options: Voice generation options.

        Returns:
            VoiceOutput: Generated audio with metadata.
        """
        pass

    @abstractmethod
    async def list_available_voices(self) -> list[Voice]:
        """
        List all available voices from this provider.

        Returns:
            list[Voice]: List of available Voice objects.
        """
        pass

    @abstractmethod
    async def process_ssml(self, ssml: str, options: Optional[VoiceOptions] = None) -> VoiceOutput:
        """
        Process SSML markup and generate voice.

        Args:
            ssml: SSML-formatted text to process.
            options: Voice generation options.

        Returns:
            VoiceOutput: Generated audio with metadata.
        """
        pass
