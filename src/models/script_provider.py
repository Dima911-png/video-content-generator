from abc import abstractmethod
from typing import Optional
from .base_provider import BaseProvider


class ScriptOptions:
    """Options for script generation."""

    def __init__(
        self,
        tone: str = "informative",
        length: str = "medium",
        style: str = "conversational",
        language: str = "en",
        **kwargs
    ):
        self.tone = tone  # e.g., 'informative', 'humorous', 'formal'
        self.length = length  # e.g., 'short', 'medium', 'long'
        self.style = style  # e.g., 'conversational', 'narrative', 'educational'
        self.language = language
        self.extra_options = kwargs


class ScriptOutput:
    """Output from script generation."""

    def __init__(self, script: str, word_count: int, metadata: Optional[dict] = None):
        self.script = script
        self.word_count = word_count
        self.metadata = metadata or {}


class IScriptProvider(BaseProvider):
    """Interface for script generation providers."""

    @abstractmethod
    async def generate_script(
        self, topic: str, options: Optional[ScriptOptions] = None
    ) -> ScriptOutput:
        """
        Generate a script based on a topic.

        Args:
            topic: The topic or concept for the script.
            options: Script generation options.

        Returns:
            ScriptOutput: Generated script with metadata.
        """
        pass

    @abstractmethod
    async def generate_outline(self, topic: str) -> list[str]:
        """
        Generate an outline/outline points for a topic.

        Args:
            topic: The topic to generate outline for.

        Returns:
            list[str]: List of outline points.
        """
        pass

    @abstractmethod
    async def enhance_script_with_ssml(self, script: str) -> str:
        """
        Enhance script with SSML tags for voice generation.

        Args:
            script: Plain text script to enhance.

        Returns:
            str: Script with SSML tags.
        """
        pass
