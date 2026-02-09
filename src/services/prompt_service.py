"""Prompt template service for script generation."""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    
    # Length guidelines (word count targets)
    length_guidelines: Dict[str, str] = field(default_factory=lambda: {
        "short": "1500-2000 words (about 8 minutes)",
        "medium": "2500-3000 words (about 10-12 minutes)",
        "long": "4000-5000 words (about 15-18 minutes)",
    })
    
    # Words per minute for duration calculations (average speaking rate)
    words_per_minute: int = 150
    
    @staticmethod
    def get_word_count_for_duration(target_minutes: float, words_per_minute: int = 150) -> str:
        """
        Calculate word count target for a specific duration.
        
        Args:
            target_minutes: Target video duration in minutes.
            words_per_minute: Speaking rate (default 150 wpm).
        
        Returns:
            Formatted string like "1500 words (about 10 minutes)"
        """
        target_words = int(target_minutes * words_per_minute)
        # Add a range (±10%)
        min_words = int(target_words * 0.9)
        max_words = int(target_words * 1.1)
        return f"{min_words}-{max_words} words (about {int(target_minutes)} minutes)"
    
    # Tone descriptions
    tone_descriptions: Dict[str, str] = field(default_factory=lambda: {
        "informative": "educational and fact-based, with clear explanations",
        "humorous": "fun and entertaining, with jokes and wit",
        "formal": "professional and polished",
        "casual": "relaxed and friendly, like chatting with a friend",
        "enthusiastic": "high-energy and excited",
    })
    
    # Style descriptions
    style_descriptions: Dict[str, str] = field(default_factory=lambda: {
        "conversational": "natural spoken language, like talking to a friend",
        "narrative": "storytelling approach with a clear arc",
        "educational": "structured teaching with clear takeaways",
        "documentary": "factual and objective presentation",
    })


class PromptService:
    """Service for building and managing prompt templates."""
    
    # Default template directory
    DEFAULT_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "prompts"
    
    def __init__(self, template_dir: Optional[str] = None, config: Optional[PromptConfig] = None):
        """
        Initialize prompt service.
        
        Args:
            template_dir: Directory containing prompt templates.
            config: Prompt configuration.
        """
        self.template_dir = Path(template_dir) if template_dir else self.DEFAULT_TEMPLATE_DIR
        self.config = config or PromptConfig()
        self._templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from the template directory."""
        if not self.template_dir.exists():
            # Create default templates if directory doesn't exist
            self._create_default_templates()
            return
        
        for template_file in self.template_dir.glob("*.txt"):
            template_name = template_file.stem
            self._templates[template_name] = template_file.read_text(encoding="utf-8")
    
    def _create_default_templates(self) -> None:
        """Create default prompt templates."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Main script generation template
        script_template = self._get_default_script_template()
        (self.template_dir / "script_generation.txt").write_text(script_template, encoding="utf-8")
        self._templates["script_generation"] = script_template
        
        # Outline template
        outline_template = self._get_default_outline_template()
        (self.template_dir / "outline.txt").write_text(outline_template, encoding="utf-8")
        self._templates["outline"] = outline_template
        
        # SSML enhancement template
        ssml_template = self._get_default_ssml_template()
        (self.template_dir / "ssml_enhancement.txt").write_text(ssml_template, encoding="utf-8")
        self._templates["ssml_enhancement"] = ssml_template
    
    def get_template(self, template_name: str) -> str:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template (without .txt extension).
            
        Returns:
            Template content string.
            
        Raises:
            ValueError: If template not found.
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[template_name]
    
    def build_script_prompt(
        self,
        topic: str,
        tone: str = "informative",
        length: str = "medium",
        style: str = "conversational",
        language: str = "en",
        research_context: str = "",
        template_name: str = "script_generation",
        custom_instructions: str = "",
        target_duration_minutes: Optional[float] = None,
    ) -> str:
        """
        Build a complete prompt for script generation.
        
        Args:
            topic: The main topic for the script.
            tone: Tone of voice (informative, humorous, etc.).
            length: Length target (short, medium, long). Ignored if target_duration_minutes is set.
            style: Writing style (conversational, narrative, etc.).
            language: Language code.
            research_context: Optional research to incorporate.
            template_name: Which template to use.
            custom_instructions: Additional custom instructions.
            target_duration_minutes: Explicit target duration in minutes (overrides length).
            
        Returns:
            Complete prompt string ready for the LLM.
        """
        template = self.get_template(template_name)
        
        # Get length guideline - prefer explicit duration if provided
        if target_duration_minutes:
            target_length = PromptConfig.get_word_count_for_duration(target_duration_minutes)
        else:
            target_length = self.config.length_guidelines.get(length, self.config.length_guidelines["medium"])
        
        # Get tone description
        tone_desc = self.config.tone_descriptions.get(tone, tone)
        
        # Get style description
        style_desc = self.config.style_descriptions.get(style, style)
        
        # Build research section
        research_section = ""
        if research_context:
            research_section = f"""
RESEARCH AND CONTEXT:
{research_context}

Use this research to support and substantiate your script. Include specific facts, statistics, and examples from the research. Do NOT include citation markers like [Citation] or [1] - this is for voice-over and citations cannot be spoken."""
        
        # Build custom instructions section
        custom_section = ""
        if custom_instructions:
            custom_section = f"""
ADDITIONAL INSTRUCTIONS:
{custom_instructions}"""
        
        # Replace placeholders in template
        prompt = template.format(
            topic=topic,
            target_length=target_length,
            tone=tone,
            tone_description=tone_desc,
            style=style,
            style_description=style_desc,
            language=language,
            research_section=research_section,
            custom_instructions=custom_section,
        )
        
        return prompt
    
    def build_outline_prompt(self, topic: str) -> str:
        """Build a prompt for outline generation."""
        template = self.get_template("outline")
        return template.format(topic=topic)
    
    def build_ssml_prompt(self, script: str) -> str:
        """Build a prompt for SSML enhancement."""
        template = self.get_template("ssml_enhancement")
        return template.format(script=script)
    
    def _get_default_script_template(self) -> str:
        """Get the default script generation template."""
        return '''Write a YouTube video script for: {topic}

{research_section}

CRITICAL INSTRUCTIONS:
This is text-to-speech content for an AI voice to read. Do NOT include:
- HOST: labels or character names (e.g., "HOST: Hey everyone")
- [Screen directions] or [visual descriptions] in brackets
- Scene markers or stage directions
- Any screenplay formatting

Write ONLY natural, flowing spoken text that a voice will read naturally.

TONE & STYLE - Make it FUN and ENGAGING:
- Write like you're chatting with a friend who's genuinely curious
- Use casual, modern language (it's okay to say "honestly", "here's the thing", "wild right?")
- Add personality! Be enthusiastic, throw in some humor where it fits
- Use rhetorical questions to keep viewers engaged ("So what does this actually mean for you?")
- Include relatable moments and pop culture references where appropriate
- Vary your energy - build excitement, then give breathing room
- Don't be afraid to be a little sarcastic or playful
- Use phrases like "let's be real", "here's where it gets interesting", "plot twist"

STRUCTURE Requirements:
- Length: {target_length} - THIS IS IMPORTANT, the video needs to hit 10+ minutes
- Tone: {tone} - {tone_description}
- Style: {style} - {style_description}
- Language: {language}
- Start with a HOOK that creates curiosity or makes a bold statement in first 10 seconds
- Tease what's coming ("and stick around because what I'm about to share will blow your mind")
- Build momentum - each section should flow naturally into the next
- Include specific examples, surprising facts, and practical takeaways
- Add "pattern interrupts" - unexpected facts or questions to re-engage attention
- Mix short punchy sentences with detailed explanations for rhythm
- Include at least 2-3 "aha moments" or surprising revelations
- Do NOT include citation markers - weave facts naturally into the narrative
- End with an impactful takeaway and clear call-to-action
- Use paragraph breaks as natural pauses for pacing

{custom_instructions}

REMEMBER: This needs to be {target_length} - expand on points, add more examples, go deeper on explanations. Don't rush through topics.

Script (plain spoken text, no formatting, no stage directions):'''
    
    def _get_default_outline_template(self) -> str:
        """Get the default outline generation template."""
        return '''Generate a concise outline for a YouTube video about: {topic}

Provide 5-7 main points as a numbered list. Be specific and actionable.
Each point should be substantial enough to discuss for 1-2 minutes.'''
    
    def _get_default_ssml_template(self) -> str:
        """Get the default SSML enhancement template."""
        return '''Take this YouTube video script and enhance it with SSML tags for better voice generation.
Add tags for:
- Natural pauses (<break time="0.5s"/>)
- Emphasis on important words (<emphasis level="strong">word</emphasis>)
- Speed variations for better pacing
- Prosody adjustments where appropriate

Keep the original text intact but wrap sections with SSML tags:

{script}

Return only the enhanced script with SSML tags, no explanations.'''


# Singleton instance for easy access
_prompt_service: Optional[PromptService] = None


def get_prompt_service(template_dir: Optional[str] = None) -> PromptService:
    """Get the prompt service singleton."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService(template_dir=template_dir)
    return _prompt_service
