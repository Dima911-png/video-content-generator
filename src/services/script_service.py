from typing import Optional
from src.models.script_provider import IScriptProvider, ScriptOptions, ScriptOutput
from src.models.schemas import ScriptRequest, ScriptResponse


class ScriptService:
    """Service orchestrating script generation workflow."""

    def __init__(self, provider: IScriptProvider):
        """
        Initialize script service with a provider.

        Args:
            provider: Script generation provider (e.g., OpenAIScriptProvider).
        """
        self.provider = provider

    async def generate_script(self, request: ScriptRequest) -> ScriptResponse:
        """
        Generate a script from a request.

        Args:
            request: ScriptRequest with topic and options.

        Returns:
            ScriptResponse with generated script and metadata.
        """
        # Validate provider
        is_valid = await self.provider.validate()
        if not is_valid:
            raise RuntimeError("Script provider is not valid or not connected")

        # Convert request to provider options
        options = ScriptOptions(
            tone=request.tone,
            length=request.length,
            style=request.style,
            language=request.language,
            **request.extra_params,
        )

        # For medium/long scripts, use multi-pass generation if available
        use_multipass = request.length in ("medium", "long") and hasattr(self.provider, 'generate_long_script')
        
        if use_multipass:
            script_output = await self.provider.generate_long_script(request.topic, options)
        else:
            script_output = await self.provider.generate_script(request.topic, options)

        # Optionally enhance with SSML tags
        script_text = script_output.script
        has_ssml_tags = request.include_ssml_tags

        if request.include_ssml_tags:
            script_text = await self.provider.enhance_script_with_ssml(script_text)
            has_ssml_tags = True

        return ScriptResponse(
            script=script_text,
            word_count=script_output.word_count,
            language=request.language,
            has_ssml_tags=has_ssml_tags,
            metadata={
                **script_output.metadata,
                **request.extra_params,
            },
        )

    async def generate_outline(self, topic: str) -> list[str]:
        """
        Generate an outline for a topic.

        Args:
            topic: Topic to generate outline for.

        Returns:
            List of outline points.
        """
        is_valid = await self.provider.validate()
        if not is_valid:
            raise RuntimeError("Script provider is not valid or not connected")

        return await self.provider.generate_outline(topic)

    async def get_provider_status(self):
        """Get the status of the underlying provider."""
        return await self.provider.get_status()

    def get_provider_capabilities(self) -> list[str]:
        """Get capabilities of the underlying provider."""
        return self.provider.get_capabilities()
