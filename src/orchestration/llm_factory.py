"""LLM factory for creating configured LangChain LLM instances."""

import os
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from .config import LLMConfig, get_config
from .logging_config import get_logger

logger = get_logger("llm_factory")


def create_llm_from_config(config: Optional[LLMConfig] = None) -> Any:
    """Create a LangChain LLM instance from configuration using init_chat_model.

    Args:
        config: LLM configuration. If None, uses global config.

    Returns:
        Configured LangChain LLM instance

    Raises:
        ValueError: If provider is not supported or API key is missing
        ImportError: If required LangChain packages are not installed
    """
    if config is None:
        config = get_config().llm

    provider = config.provider.lower()

    # Check API key is available
    api_key_env_var = config.api_key_env_var or get_default_api_key_env_var(provider)
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_key_env_var}"
        )

    # Create the model identifier for init_chat_model
    # Users should provide correct provider names as per LangChain documentation
    model_id = f"{provider}:{config.model_name}"

    logger.info(
        f"Creating LLM instance: provider={provider}, model={config.model_name}"
    )

    try:
        # Use init_chat_model with our configuration
        kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        kwargs.update(config.additional_params)

        return init_chat_model(model=model_id, **kwargs)

    except Exception as e:
        error_msg = (
            f"Failed to create LLM instance with provider '{provider}' and model '{config.model_name}': {e}\n"
            f"Please ensure:\n"
            f"1. The provider name '{provider}' is correct for LangChain init_chat_model\n"
            f"2. The model name '{config.model_name}' is supported by the provider\n"
            f"3. Required LangChain integration packages are installed\n"
            f"4. API key is set in environment variable: {api_key_env_var}\n"
            f"For supported provider formats, see: https://python.langchain.com/docs/integrations/chat/"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def get_default_api_key_env_var(provider: str) -> str:
    """Get default API key environment variable for a provider.

    Args:
        provider: LLM provider name

    Returns:
        Environment variable name for API key
    """
    defaults = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    return defaults.get(provider.lower(), f"{provider.upper()}_API_KEY")


def get_supported_providers() -> list[str]:
    """Get list of commonly used LLM providers.

    Note: This is not exhaustive - LangChain init_chat_model supports many providers.
    For the complete list, see: https://python.langchain.com/docs/integrations/chat/

    Returns:
        List of commonly used provider names
    """
    return ["anthropic", "openai", "google_genai", "azure_openai", "ollama", "cohere"]


def validate_llm_config(config: LLMConfig) -> list[str]:
    """Validate LLM configuration and return any issues found.

    Args:
        config: LLM configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check provider (informational warning, not an error)
    common_providers = get_supported_providers()
    if config.provider.lower() not in common_providers:
        issues.append(
            f"Note: Provider '{config.provider}' is not in the list of commonly used providers: {', '.join(common_providers)}. "
            f"If this is a valid LangChain provider, this is not an error. "
            f"For all supported providers, see: https://python.langchain.com/docs/integrations/chat/"
        )

    # Check temperature range
    if not 0.0 <= config.temperature <= 1.0:
        issues.append(
            f"Temperature must be between 0.0 and 1.0, got: {config.temperature}"
        )

    # Check max_tokens
    if config.max_tokens <= 0:
        issues.append(f"max_tokens must be positive, got: {config.max_tokens}")

    # Check timeout
    if config.timeout <= 0:
        issues.append(f"timeout must be positive, got: {config.timeout}")

    # Check API key environment variable exists
    api_key_env_var = config.api_key_env_var or get_default_api_key_env_var(
        config.provider
    )
    if not os.getenv(api_key_env_var):
        issues.append(f"API key not found in environment variable: {api_key_env_var}")

    return issues
