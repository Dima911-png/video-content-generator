import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from src.models.schemas import ProviderConfig


class ProvidersConfig:
    """Configuration manager for providers supporting both .env and settings.json."""

    def __init__(self, env_path: Optional[str] = None, settings_path: Optional[str] = None):
        """
        Initialize configuration from .env and settings.json files.

        Args:
            env_path: Path to .env file (defaults to project root)
            settings_path: Path to settings.json (defaults to config/settings.json)
        """
        self.env_path = env_path or Path(__file__).parent.parent / ".env"
        self.settings_path = settings_path or Path(__file__).parent / "settings.json"

        # Load environment variables
        if os.path.exists(self.env_path):
            load_dotenv(self.env_path)

        # Load settings from JSON
        self.settings: Dict[str, Any] = {}
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r") as f:
                self.settings = json.load(f)

    def get_provider_config(self, provider_type: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.

        Args:
            provider_type: Type of provider (e.g., 'openai', 'elevenlabs')

        Returns:
            ProviderConfig: Configuration object for the provider
        """
        # Start with settings from JSON
        provider_settings = self.settings.get("providers", {}).get(provider_type, {})

        # Override with environment variables (env vars take precedence)
        api_key = os.getenv(f"{provider_type.upper()}_API_KEY")
        if api_key:
            provider_settings["api_key"] = api_key

        # Handle other common env vars
        api_url = os.getenv(f"{provider_type.upper()}_API_URL")
        if api_url:
            provider_settings["api_url"] = api_url

        # Ensure provider_type is set
        provider_settings["provider_type"] = provider_type

        return ProviderConfig(**provider_settings)

    def get_script_provider_type(self) -> str:
        """Get the configured script provider type."""
        return os.getenv("SCRIPT_PROVIDER", self.settings.get("script_provider", "openai"))

    def get_voice_provider_type(self) -> str:
        """Get the configured voice provider type."""
        return os.getenv("VOICE_PROVIDER", self.settings.get("voice_provider", "elevenlabs"))

    def get_script_provider_config(self) -> ProviderConfig:
        """Get configuration for the script provider."""
        provider_type = self.get_script_provider_type()
        return self.get_provider_config(provider_type)

    def get_voice_provider_config(self) -> ProviderConfig:
        """Get configuration for the voice provider."""
        provider_type = self.get_voice_provider_type()
        return self.get_provider_config(provider_type)

    def get_general_config(self) -> Dict[str, Any]:
        """Get general configuration settings."""
        return self.settings.get("general", {})

    def get_image_api_keys(self) -> Dict[str, str]:
        """
        Get API keys for image providers (Pexels, Pixabay).
        
        Returns:
            Dictionary with API keys for image services.
        """
        keys = {}
        
        # Pexels API key
        pexels_key = os.getenv("PEXELS_API_KEY")
        if pexels_key and not pexels_key.startswith("your_"):
            keys["pexels"] = pexels_key
        
        # Pixabay API key
        pixabay_key = os.getenv("PIXABAY_API_KEY")
        if pixabay_key and not pixabay_key.startswith("your_"):
            keys["pixabay"] = pixabay_key
        
        return keys

    def get_timeout(self, provider_type: str = None) -> int:
        """Get timeout setting for a provider or default."""
        if provider_type:
            config = self.get_provider_config(provider_type)
            return config.timeout
        return self.settings.get("general", {}).get("timeout", 30)

    def get_max_retries(self, provider_type: str = None) -> int:
        """Get max retries setting for a provider or default."""
        if provider_type:
            config = self.get_provider_config(provider_type)
            return config.max_retries
        return self.settings.get("general", {}).get("max_retries", 3)
