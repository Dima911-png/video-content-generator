from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    provider_type: str = Field(..., description="Type of provider (e.g., 'openai', 'elevenlabs')")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    api_url: Optional[str] = Field(None, description="Custom API URL if applicable")
    model_name: Optional[str] = Field(None, description="Model or engine to use")
    timeout: int = Field(120, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    class Config:
        extra = "allow"


class ScriptRequest(BaseModel):
    """Request for script generation."""

    topic: str = Field(..., description="The topic or concept for the script")
    tone: str = Field("informative", description="Tone of the script (e.g., 'informative', 'humorous')")
    length: str = Field("medium", description="Length of the script (e.g., 'short', 'medium', 'long')")
    style: str = Field("conversational", description="Style of the script")
    language: str = Field("en", description="Language code (e.g., 'en', 'es', 'fr')")
    include_ssml_tags: bool = Field(True, description="Whether to include SSML tags in output")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    class Config:
        extra = "allow"


class ScriptResponse(BaseModel):
    """Response from script generation."""

    script: str = Field(..., description="The generated script text")
    word_count: int = Field(..., description="Number of words in the script")
    language: str = Field("en", description="Language of the script")
    has_ssml_tags: bool = Field(False, description="Whether script contains SSML tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VoiceRequest(BaseModel):
    """Request for voice generation."""

    text: str = Field(..., description="Text to convert to speech (can include SSML)")
    voice_id: Optional[str] = Field(None, description="ID of the voice to use")
    speed: float = Field(1.0, description="Speech speed multiplier (typically 0.5-2.0)")
    pitch: float = Field(1.0, description="Pitch multiplier (typically 0.5-2.0)")
    language: str = Field("en", description="Language code")
    format: str = Field("mp3", description="Output audio format (e.g., 'mp3', 'wav')")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    class Config:
        extra = "allow"


class VoiceResponse(BaseModel):
    """Response from voice generation."""

    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio data")
    format: str = Field("mp3", description="Audio format")
    duration_seconds: Optional[float] = Field(None, description="Duration of audio in seconds")
    voice_id: Optional[str] = Field(None, description="Voice ID used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Voice(BaseModel):
    """Represents an available voice."""

    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    language: str = Field("en", description="Language of the voice")
    gender: Optional[str] = Field(None, description="Gender if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        extra = "allow"


class ContentRequest(BaseModel):
    """High-level request for content generation."""

    topic: str = Field(..., description="The topic for content generation")
    script_request: ScriptRequest = Field(..., description="Script generation parameters")
    voice_request: VoiceRequest = Field(..., description="Voice generation parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentResponse(BaseModel):
    """Response from complete content generation pipeline."""

    script_response: ScriptResponse = Field(..., description="Generated script")
    voice_response: VoiceResponse = Field(..., description="Generated voice")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline metadata")
