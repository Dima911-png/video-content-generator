import re
from typing import List, Dict, Any


class SSMLTag:
    """Represents an SSML tag."""

    def __init__(self, tag_type: str, content: str, attributes: Dict[str, str] = None):
        self.tag_type = tag_type
        self.content = content
        self.attributes = attributes or {}

    def __repr__(self):
        return f"SSMLTag({self.tag_type}, {self.content[:30]}...)"


class SsmlProcessor:
    """Processor for handling SSML tags in scripts."""

    # Common SSML tags
    VALID_TAGS = {
        "speak",
        "p",
        "s",
        "break",
        "emphasis",
        "prosody",
        "say-as",
        "sub",
        "phoneme",
        "amazon:effect",
    }

    @staticmethod
    def is_valid_ssml(text: str) -> bool:
        """
        Check if text contains valid SSML markup.

        Args:
            text: Text to validate.

        Returns:
            bool: True if text contains valid SSML tags.
        """
        if not text:
            return False

        # Must have opening speak tag
        if "<speak>" not in text or "</speak>" not in text:
            return False

        # Check for unmatched tags
        return SsmlProcessor._check_balanced_tags(text)

    @staticmethod
    def _check_balanced_tags(text: str) -> bool:
        """Check if SSML tags are balanced."""
        tag_stack = []
        pattern = r"</?(\w+)[^>]*>"

        for match in re.finditer(pattern, text):
            tag = match.group(1)
            if tag in ["break", "amazon:effect"]:  # Self-closing tags
                continue
            if match.group(0).startswith("</"):
                if not tag_stack or tag_stack[-1] != tag:
                    return False
                tag_stack.pop()
            else:
                tag_stack.append(tag)

        return len(tag_stack) == 0

    @staticmethod
    def wrap_with_speak(text: str) -> str:
        """
        Wrap text with <speak> tags if not already wrapped.

        Args:
            text: Text to wrap.

        Returns:
            str: Text wrapped with speak tags.
        """
        if "<speak>" in text and "</speak>" in text:
            return text
        return f"<speak>{text}</speak>"

    @staticmethod
    def add_break(text: str, time: str = "0.5s", position: str = "after") -> str:
        """
        Add a break tag to text.

        Args:
            text: Text to add break to.
            time: Break duration (e.g., "0.5s", "500ms").
            position: 'before', 'after', or 'both'.

        Returns:
            str: Text with break tags added.
        """
        break_tag = f'<break time="{time}"/>'

        if position == "before":
            return f"{break_tag}{text}"
        elif position == "after":
            return f"{text}{break_tag}"
        elif position == "both":
            return f"{break_tag}{text}{break_tag}"
        return text

    @staticmethod
    def add_emphasis(text: str, level: str = "moderate") -> str:
        """
        Add emphasis to text.

        Args:
            text: Text to emphasize.
            level: 'strong', 'moderate', or 'reduced'.

        Returns:
            str: Text wrapped with emphasis tag.
        """
        return f'<emphasis level="{level}">{text}</emphasis>'

    @staticmethod
    def add_prosody(
        text: str,
        rate: str = None,
        pitch: str = None,
        volume: str = None,
    ) -> str:
        """
        Add prosody modifications to text.

        Args:
            text: Text to modify.
            rate: Speech rate (e.g., "slow", "1.2", "fast").
            pitch: Pitch level (e.g., "high", "+50%", "-10%").
            volume: Volume level (e.g., "loud", "+6dB", "-3dB").

        Returns:
            str: Text wrapped with prosody tag.
        """
        attributes = []
        if rate:
            attributes.append(f'rate="{rate}"')
        if pitch:
            attributes.append(f'pitch="{pitch}"')
        if volume:
            attributes.append(f'volume="{volume}"')

        attr_str = " ".join(attributes)
        return f"<prosody {attr_str}>{text}</prosody>"

    @staticmethod
    def add_pause_between_sentences(text: str, pause_time: str = "1s") -> str:
        """
        Add breaks between sentences.

        Args:
            text: Text to add pauses to.
            pause_time: Pause duration between sentences.

        Returns:
            str: Text with pauses added after sentences.
        """
        # Split by sentence endings and add breaks
        break_tag = f'<break time="{pause_time}"/>'
        sentences = re.split(r"([.!?])", text)

        result = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                result.append(sentences[i])
            if i + 1 < len(sentences):
                result.append(sentences[i + 1])
                result.append(break_tag)

        return "".join(result).rstrip(break_tag)

    @staticmethod
    def remove_ssml_tags(text: str) -> str:
        """
        Remove all SSML tags from text.

        Args:
            text: Text with SSML tags.

        Returns:
            str: Plain text without SSML tags.
        """
        # Remove all XML-like tags
        clean_text = re.sub(r"<[^>]+>", "", text)
        return clean_text.strip()

    @staticmethod
    def extract_ssml_tags(text: str) -> List[SSMLTag]:
        """
        Extract all SSML tags from text.

        Args:
            text: Text containing SSML tags.

        Returns:
            List[SSMLTag]: List of extracted SSML tags.
        """
        tags = []
        pattern = r"<(\w+)([^>]*)>([^<]*)</\1>|<(\w+)([^>]*)/>"

        for match in re.finditer(pattern, text):
            if match.group(1):  # Opening/closing tag pair
                tag_type = match.group(1)
                attributes_str = match.group(2).strip()
                content = match.group(3)
            else:  # Self-closing tag
                tag_type = match.group(4)
                attributes_str = match.group(5).strip()
                content = ""

            # Parse attributes
            attributes = {}
            for attr_match in re.finditer(r'(\w+)="([^"]*)"', attributes_str):
                attributes[attr_match.group(1)] = attr_match.group(2)

            tags.append(SSMLTag(tag_type, content, attributes))

        return tags

    @staticmethod
    def validate_ssml_structure(text: str) -> tuple[bool, str]:
        """
        Validate SSML structure and return errors.

        Args:
            text: SSML text to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not text:
            return False, "Text is empty"

        if "<speak>" not in text:
            return False, "Missing <speak> opening tag"

        if "</speak>" not in text:
            return False, "Missing </speak> closing tag"

        if not SsmlProcessor._check_balanced_tags(text):
            return False, "Unbalanced tags detected"

        # Check for valid tag names
        invalid_tags = re.findall(r"<(\w+)[^>]*>", text)
        for tag in set(invalid_tags):
            if tag not in SsmlProcessor.VALID_TAGS and not tag.startswith("amazon:"):
                return False, f"Invalid SSML tag: {tag}"

        return True, "Valid SSML"
