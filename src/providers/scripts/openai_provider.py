"""OpenAI script generation provider.

This module provides script generation functionality using the OpenAI API,
supporting multiple models (GPT-4, GPT-4 Turbo) and customizable prompts.
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


class OpenAIScriptProvider(IScriptProvider):
    """Script generation provider using OpenAI API."""

    def __init__(self, config: ProviderConfig, prompt_service: Optional[PromptService] = None):
        """
        Initialize OpenAI script provider.

        Args:
            config: Provider configuration with API key and settings.
            prompt_service: Optional prompt service for building prompts.
        """
        self.config = config
        self.api_key = config.api_key
        self.api_url = config.api_url or "https://api.openai.com/v1"
        self.model = config.model_name or "gpt-4"
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.extra_params = config.extra_params or {}
        
        # Use provided prompt service or get singleton
        self.prompt_service = prompt_service or get_prompt_service()

    async def validate(self) -> bool:
        """Validate OpenAI configuration and connectivity."""
        if not self.api_key:
            return False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.get(
                f"{self.api_url}/models/{self.model}",
                headers=headers,
                timeout=self.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def get_status(self) -> ProviderStatus:
        """Get current status of OpenAI provider."""
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
        """Generate a script using OpenAI API."""
        if not options:
            options = ScriptOptions()

        # Check for explicit target duration from extra_options
        target_duration_minutes = options.extra_options.get("target_duration_minutes")

        # Use prompt service to build the prompt
        prompt = self.prompt_service.build_script_prompt(
            topic=topic,
            tone=options.tone,
            length=options.length,
            style=options.style,
            language=options.language,
            research_context=options.extra_options.get("research_context", ""),
            template_name=options.extra_options.get("template_name", "script_generation"),
            custom_instructions=options.extra_options.get("custom_instructions", ""),
            target_duration_minutes=target_duration_minutes,
        )

        # Get the target length description - use explicit duration if provided
        if target_duration_minutes:
            from src.services.prompt_service import PromptConfig
            target_length = PromptConfig.get_word_count_for_duration(target_duration_minutes)
        else:
            target_length = self.prompt_service.config.length_guidelines.get(
                options.length, "2500-3000 words"
            )

        # System message strongly emphasizing length requirement
        system_message = f"""You are an expert YouTube scriptwriter who creates engaging, detailed content.

CRITICAL LENGTH REQUIREMENT: Your script MUST be {target_length}. This is non-negotiable.

To achieve this length you must:
- Expand each main point with detailed explanations, examples, and analogies
- Include multiple specific examples for every concept
- Add interesting tangents and related facts
- Use storytelling and anecdotes to illustrate points
- Never summarize or rush through topics - go deep
- Include transitions and "pattern interrupts" between sections

Count your output mentally - you need approximately 2500-3000 words minimum.
A script under 2000 words is a FAILURE. Aim for the higher end of the range."""

        for attempt in range(self.max_retries):
            try:
                response = await self._call_openai(prompt, system_message=system_message)
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

    async def generate_long_script(
        self, topic: str, options: Optional[ScriptOptions] = None
    ) -> ScriptOutput:
        """
        Generate a long script using multi-pass approach.
        
        This generates an outline first, then expands each section separately,
        which produces much longer and more detailed content.
        """
        if not options:
            options = ScriptOptions()

        target_length = self.prompt_service.config.length_guidelines.get(
            options.length, "2500-3000 words"
        )

        # Step 1: Generate detailed outline with 6-8 sections
        outline_prompt = f"""Create a detailed outline for a YouTube video about: {topic}

The final video needs to be {target_length} (about 10-12 minutes of speaking).

Create 6-8 main sections. For each section, list:
- Section title
- 3-4 key points to cover
- An interesting example or story idea

Format:
SECTION 1: [Title]
- Point 1
- Point 2  
- Point 3
- Example/Story idea

(continue for all sections)"""

        outline_response = await self._call_openai(outline_prompt)
        
        # Step 2: Generate intro hook (200-250 words)
        intro_prompt = f"""Write a SHORT, engaging introduction/hook for a YouTube video about: {topic}

This should be exactly 200-250 words (no more!) and include:
- An attention-grabbing opening statement or question
- Tease what viewers will learn
- Use casual, fun language

Write ONLY the spoken text - no stage directions, no [brackets], no "HOST:" labels.

Just natural flowing speech. Keep it SHORT - 200-250 words max."""

        intro = await self._call_openai(intro_prompt)

        # Step 3: Expand each section (400-500 words each)
        sections = []
        section_titles = re.findall(r'SECTION \d+:\s*(.+?)(?:\n|$)', outline_response)
        
        if not section_titles:
            # Fallback: split outline into chunks
            section_titles = ["Introduction and Background", "Key Features and Capabilities", 
                            "Practical Applications", "Benefits and Advantages",
                            "Challenges and Considerations", "Future Outlook and Tips"]

        for i, title in enumerate(section_titles[:4]):  # Limit to 4 sections for ~10 min video
            section_prompt = f"""You are writing section {i+1} of a YouTube video about: {topic}

This section is about: {title}

Write exactly 300-350 words for this section (no more!). Requirements:
- Use casual, engaging, fun language 
- Include 1-2 specific examples
- Be concise and punchy
- Transition naturally
- Write for text-to-speech - no brackets, no stage directions

Write ONLY the spoken text (300-350 words max):"""

            section_content = await self._call_openai(section_prompt)
            sections.append(section_content.strip())

        # Step 4: Generate conclusion (150-200 words)
        conclusion_prompt = f"""Write a SHORT conclusion for a YouTube video about: {topic}

This should be exactly 150-200 words (no more!) and include:
- Quick summary of key points
- Call-to-action (like, subscribe, comment)
- End on an upbeat note

Write ONLY the spoken text. Keep it SHORT - 150-200 words max."""

        conclusion = await self._call_openai(conclusion_prompt)

        # Combine all parts
        full_script = intro.strip() + "\n\n" + "\n\n".join(sections) + "\n\n" + conclusion.strip()
        word_count = len(full_script.split())

        return ScriptOutput(
            script=full_script,
            word_count=word_count,
            metadata={
                "model": self.model,
                "tone": options.tone,
                "length": options.length,
                "language": options.language,
                "generation_method": "multi-pass",
                "sections_generated": len(sections) + 2,  # intro + sections + conclusion
            },
        )

    async def generate_outline(self, topic: str) -> List[str]:
        """Generate outline points for a topic."""
        prompt = self.prompt_service.build_outline_prompt(topic)
        response = await self._call_openai(prompt)

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
        """Enhance script with SSML tags using OpenAI."""
        # For long scripts, process in chunks to avoid token limits
        paragraphs = script.split('\n\n')
        
        if len(paragraphs) > 5:
            # Process in chunks of 3-4 paragraphs
            enhanced_parts = []
            chunk_size = 3
            
            for i in range(0, len(paragraphs), chunk_size):
                chunk = '\n\n'.join(paragraphs[i:i + chunk_size])
                prompt = self.prompt_service.build_ssml_prompt(chunk)
                enhanced_chunk = await self._call_openai(prompt)
                enhanced_parts.append(self._strip_markdown_code_fences(enhanced_chunk))
            
            return '\n\n'.join(enhanced_parts)
        else:
            # Short script - process all at once
            prompt = self.prompt_service.build_ssml_prompt(script)
            enhanced = await self._call_openai(prompt)
            return self._strip_markdown_code_fences(enhanced)

    async def _call_openai(self, prompt: str, system_message: str = None) -> str:
        """Call OpenAI API and return the response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.extra_params.get("temperature", 0.8),
            "max_tokens": self.extra_params.get("max_tokens", 4500),
        }

        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
