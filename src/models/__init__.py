"""Models and interfaces for providers."""

from .base_provider import BaseProvider, ProviderStatus
from .script_provider import IScriptProvider, ScriptOptions, ScriptOutput
from .voice_provider import IVoiceProvider, VoiceOptions, VoiceOutput, Voice
from .schemas import (
    ProviderConfig,
    ScriptRequest,
    ScriptResponse,
    VoiceRequest,
    VoiceResponse,
    Voice as VoiceSchema,
    ContentRequest,
    ContentResponse,
)

__all__ = [
    "BaseProvider",
    "ProviderStatus",
    "IScriptProvider",
    "ScriptOptions",
    "ScriptOutput",
    "IVoiceProvider",
    "VoiceOptions",
    "VoiceOutput",
    "Voice",
    "ProviderConfig",
    "ScriptRequest",
    "ScriptResponse",
    "VoiceRequest",
    "VoiceResponse",
    "VoiceSchema",
    "ContentRequest",
    "ContentResponse",
]
