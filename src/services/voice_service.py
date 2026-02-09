import base64
from typing import Optional
from src.models.voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput
from src.models.schemas import VoiceRequest, VoiceResponse


class VoiceService:
    """Service orchestrating voice generation workflow."""

    def __init__(self, provider: IVoiceProvider):
        """
        Initialize voice service with a provider.

        Args:
            provider: Voice generation provider (e.g., ElevenLabsVoiceProvider).
        """
        self.provider = provider

    async def generate_voice(self, request: VoiceRequest) -> VoiceResponse:
        """
        Generate voice from a request.

        Args:
            request: VoiceRequest with text and voice options.

        Returns:
            VoiceResponse with generated audio (base64 encoded) and metadata.
        """
        # Validate provider
        is_valid = await self.provider.validate()
        if not is_valid:
            raise RuntimeError("Voice provider is not valid or not connected")

        # Convert request to provider options
        options = VoiceOptions(
            voice_id=request.voice_id,
            speed=request.speed,
            pitch=request.pitch,
            language=request.language,
            **request.extra_params,
        )

        # Determine if SSML is present in text
        has_ssml = "<speak>" in request.text or "<prosody>" in request.text

        # Generate voice
        if has_ssml:
            voice_output = await self.provider.process_ssml(request.text, options)
        else:
            voice_output = await self.provider.generate_voice(request.text, options)

        # Encode audio to base64
        audio_base64 = base64.b64encode(voice_output.audio).decode("utf-8")

        return VoiceResponse(
            audio_base64=audio_base64,
            format=request.format,
            voice_id=request.voice_id,
            metadata={
                **voice_output.metadata,
                **request.extra_params,
            },
        )

    async def list_voices(self):
        """List all available voices from the provider."""
        is_valid = await self.provider.validate()
        if not is_valid:
            raise RuntimeError("Voice provider is not valid or not connected")

        return await self.provider.list_available_voices()

    async def get_provider_status(self):
        """Get the status of the underlying provider."""
        return await self.provider.get_status()

    def get_provider_capabilities(self) -> list[str]:
        """Get capabilities of the underlying provider."""
        return self.provider.get_capabilities()
