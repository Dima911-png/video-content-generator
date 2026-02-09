from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ProviderStatus:
    """Represents the status of a provider."""

    def __init__(self, is_available: bool, message: str = ""):
        self.is_available = is_available
        self.message = message


class BaseProvider(ABC):
    """Abstract base class for all content generation providers."""

    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate provider configuration and connectivity.

        Returns:
            bool: True if provider is valid and accessible, False otherwise.
        """
        pass

    @abstractmethod
    async def get_status(self) -> ProviderStatus:
        """
        Get the current status of the provider.

        Returns:
            ProviderStatus: Status object containing availability and message.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this provider supports.

        Returns:
            List[str]: List of capability strings (e.g., ['generate_script', 'multilingual']).
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current provider configuration (without sensitive keys).

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        pass
