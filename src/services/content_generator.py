from typing import Optional
from src.services.script_service import ScriptService
from src.services.voice_service import VoiceService
from src.models.schemas import (
    ContentRequest,
    ContentResponse,
    ScriptRequest,
    VoiceRequest,
)


class ContentGenerator:
    """Main pipeline orchestrator for generating YouTube content (script + voice)."""

    def __init__(self, script_service: ScriptService, voice_service: VoiceService):
        """
        Initialize content generator with services.

        Args:
            script_service: Service for script generation.
            voice_service: Service for voice generation.
        """
        self.script_service = script_service
        self.voice_service = voice_service

    async def generate_content(self, request: ContentRequest) -> ContentResponse:
        """
        Generate complete content: script + voice.

        Args:
            request: ContentRequest with topic and generation options.

        Returns:
            ContentResponse with generated script and voice.
        """
        # Validate both services
        script_status = await self.script_service.get_provider_status()
        voice_status = await self.voice_service.get_provider_status()

        if not script_status.is_available or not voice_status.is_available:
            raise RuntimeError(
                f"Services not available. Script: {script_status.message}, "
                f"Voice: {voice_status.message}"
            )

        # Generate script
        script_response = await self.script_service.generate_script(request.script_request)

        # Use generated script as voice input
        voice_request = request.voice_request
        voice_request.text = script_response.script  # Use the generated script

        # Generate voice
        voice_response = await self.voice_service.generate_voice(voice_request)

        # Combine responses
        return ContentResponse(
            script_response=script_response,
            voice_response=voice_response,
            metadata={
                **request.metadata,
                "pipeline": "script_to_voice",
                "script_word_count": script_response.word_count,
                "voice_format": voice_response.format,
            },
        )

    async def generate_script_only(self, request: ContentRequest) -> object:
        """Generate only the script."""
        return await self.script_service.generate_script(request.script_request)

    async def generate_voice_only(self, text: str, voice_request: VoiceRequest) -> object:
        """Generate only voice from provided text."""
        voice_request.text = text
        return await self.voice_service.generate_voice(voice_request)

    async def get_service_status(self) -> dict:
        """Get status of both services."""
        script_status = await self.script_service.get_provider_status()
        voice_status = await self.voice_service.get_provider_status()

        return {
            "script_service": {
                "available": script_status.is_available,
                "message": script_status.message,
                "capabilities": self.script_service.get_provider_capabilities(),
            },
            "voice_service": {
                "available": voice_status.is_available,
                "message": voice_status.message,
                "capabilities": self.voice_service.get_provider_capabilities(),
            },
        }
