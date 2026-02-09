"""Output manager for saving generated content."""

import base64
import json
from pathlib import Path
from datetime import datetime
from src.models.schemas import ContentResponse, ScriptResponse, VoiceResponse


class OutputManager:
    """Manages saving generated content to disk."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize output manager.

        Args:
            output_dir: Directory to save output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_content(
        self,
        topic: str,
        script_response: ScriptResponse,
        voice_response: VoiceResponse,
    ) -> dict:
        """
        Save script and voice to disk.

        Args:
            topic: Topic name for naming files.
            script_response: Generated script.
            voice_response: Generated voice.

        Returns:
            dict: Paths to saved files.
        """
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:50]
        content_dir = self.output_dir / f"{timestamp}_{topic_slug}"
        content_dir.mkdir(exist_ok=True)

        # Save script
        script_path = content_dir / "script.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_response.script)

        # Save script metadata
        metadata_path = content_dir / "script_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "word_count": script_response.word_count,
                    "language": script_response.language,
                    "has_ssml_tags": script_response.has_ssml_tags,
                    "metadata": script_response.metadata,
                },
                f,
                indent=2,
            )

        # Decode and save audio
        audio_data = base64.b64decode(voice_response.audio_base64)
        audio_path = content_dir / f"voice.{voice_response.format}"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # Save voice metadata
        voice_metadata_path = content_dir / "voice_metadata.json"
        with open(voice_metadata_path, "w") as f:
            json.dump(
                {
                    "format": voice_response.format,
                    "voice_id": voice_response.voice_id,
                    "audio_size_bytes": len(audio_data),
                    "metadata": voice_response.metadata,
                },
                f,
                indent=2,
            )

        # Save combined metadata
        combined_path = content_dir / "content_metadata.json"
        with open(combined_path, "w") as f:
            json.dump(
                {
                    "topic": topic,
                    "timestamp": timestamp,
                    "script_file": str(script_path.relative_to(self.output_dir)),
                    "audio_file": str(audio_path.relative_to(self.output_dir)),
                    "script_word_count": script_response.word_count,
                    "audio_size_bytes": len(audio_data),
                },
                f,
                indent=2,
            )

        return {
            "script": str(script_path),
            "audio": str(audio_path),
            "metadata": str(combined_path),
            "directory": str(content_dir),
        }

    def save_script_only(self, topic: str, script_response: ScriptResponse) -> str:
        """Save only a script."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:50]
        content_dir = self.output_dir / f"{timestamp}_{topic_slug}"
        content_dir.mkdir(exist_ok=True)

        script_path = content_dir / "script.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_response.script)

        return str(script_path)

    def save_voice_only(self, topic: str, voice_response: VoiceResponse) -> str:
        """Save only voice/audio."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = topic.lower().replace(" ", "_")[:50]
        content_dir = self.output_dir / f"{timestamp}_{topic_slug}"
        content_dir.mkdir(exist_ok=True)

        audio_data = base64.b64decode(voice_response.audio_base64)
        audio_path = content_dir / f"voice.{voice_response.format}"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        return str(audio_path)
