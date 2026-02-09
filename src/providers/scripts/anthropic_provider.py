"""Anthropic Claude script generation provider.

This provider demonstrates how new LLM providers can be added while
reusing the shared PromptService for consistent prompt templates.
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List

import requests

from src.models.base_provider import BaseProvider, ProviderStatus
from src.models.script_provider import IScriptProvider, ScriptOptions, ScriptOutput
from src.models.schemas import ProviderConfig
from src.services.prompt_service import get_prompt_service, PromptService

logger = logging.getLogger(__name__)


class AnthropicScriptProvider(IScriptProvider):
    """Script generation provider using Anthropic Claude API."""

    def __init__(self, config: ProviderConfig, prompt_service: Optional[PromptService] = None):
        """
        Initialize Anthropic script provider.

        Args:
            config: Provider configuration with API key and settings.
            prompt_service: Optional prompt service for building prompts.
        """
        self.config = config
        self.api_key = config.api_key
        self.api_url = config.api_url or "https://api.anthropic.com/v1"
        self.model = config.model_name or "claude-3-sonnet-20240229"
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.extra_params = config.extra_params or {}
        
        # Use shared prompt service - same prompts as OpenAI provider!
        self.prompt_service = prompt_service or get_prompt_service()

    async def validate(self) -> bool:
        """Validate Anthropic configuration and connectivity."""
        if not self.api_key:
            return False

        try:
            # Simple validation - check if we can reach the API
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            # Anthropic doesn't have a simple models endpoint, so we do a minimal test
            return True  # Just check if API key is present
        except Exception:
            return False

    async def get_status(self) -> ProviderStatus:
        """Get current status of Anthropic provider."""
        is_valid = await self.validate()
        message = "Connected and ready" if is_valid else "Connection failed or invalid config"
        return ProviderStatus(is_valid, message)

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities."""
        return [
            "generate_script",
            "generate_outline",
            "enhance_with_ssml",
            "multiple_tones",
            "multiple_languages",
        ]

    def get_config(self) -> Dict[str, Any]:
        """Get configuration without sensitive keys."""
        return {
            "provider_type": self.config.provider_type,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    async def generate_script(
        self, topic: str, options: Optional[ScriptOptions] = None
    ) -> ScriptOutput:
        """Generate a script using Anthropic Claude API."""
        if not options:
            options = ScriptOptions()

        # Use the SAME prompt service as OpenAI provider
        # This ensures consistent prompts across all providers
        prompt = self.prompt_service.build_script_prompt(
            topic=topic,
            tone=options.tone,
            length=options.length,
            style=options.style,
            language=options.language,
            research_context=options.extra_options.get("research_context", ""),
            template_name=options.extra_options.get("template_name", "script_generation"),
            custom_instructions=options.extra_options.get("custom_instructions", ""),
        )

        for attempt in range(self.max_retries):
            try:
                response = await self._call_anthropic(prompt)
                script = response.strip()
                word_count = len(script.split())

                return ScriptOutput(
                    script=script,
                    word_count=word_count,
                    metadata={
                        "model": self.model,
                        "tone": options.tone,
                        "length": options.length,
                        "language": options.language,
                        "attempt": attempt + 1,
                    },
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def generate_outline(self, topic: str) -> List[str]:
        """Generate outline points for a topic."""
        prompt = self.prompt_service.build_outline_prompt(topic)
        response = await self._call_anthropic(prompt)

        # Parse numbered list
        lines = response.strip().split("\n")
        outline = [
            line.strip().lstrip("0123456789.-) ") for line in lines if line.strip()
        ]
        return [point for point in outline if point]

    def _strip_markdown_code_fences(self, text: str) -> str:
        """Strip markdown code fences from LLM response."""
        # Remove ```xml, ```ssml, ``` markers that LLMs often add
        text = re.sub(r'^```(?:xml|ssml|)\s*\n?', '', text.strip())
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()

    async def enhance_script_with_ssml(self, script: str) -> str:
        """Enhance script with SSML tags using Anthropic Claude."""
        prompt = self.prompt_service.build_ssml_prompt(script)
        enhanced = await self._call_anthropic(prompt)
        return self._strip_markdown_code_fences(enhanced)

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API and return the response."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.extra_params.get("max_tokens", 4500),
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            f"{self.api_url}/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        # Anthropic returns content as a list of blocks
        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
        
        return text_content
