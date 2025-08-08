"""LLM factory for creating configured LangChain LLM instances."""

import os
from typing import Any, Dict, Optional

from .config import LLMConfig, get_config
from .logging_config import get_logger

logger = get_logger("llm_factory")


def create_llm_from_config(config: Optional[LLMConfig] = None) -> Any:
    """Create a LangChain LLM instance from configuration.

    Args:
        config: LLM configuration. If None, uses global config.

    Returns:
        Configured LangChain LLM instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required LangChain packages are not installed
    """
    if config is None:
        config = get_config().llm

    provider = config.provider.lower()

    # Base parameters common to most providers
    params = {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        **config.additional_params,
    }

    # Add base_url if specified
    if config.base_url:
        params["base_url"] = config.base_url

    logger.info(
        f"Creating LLM instance: provider={provider}, model={config.model_name}"
    )

    if provider == "anthropic":
        return _create_anthropic_llm(config, params)
    elif provider == "openai":
        return _create_openai_llm(config, params)
    elif provider == "google":
        return _create_google_llm(config, params)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_anthropic_llm(config: LLMConfig, params: Dict[str, Any]) -> Any:
    """Create Anthropic Claude LLM instance."""
    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore
    except ImportError:
        raise ImportError(
            "langchain_anthropic package is required for Anthropic provider. "
            "Install with: pip install langchain-anthropic"
        )

    # Get API key from environment
    api_key_env_var = config.api_key_env_var or "ANTHROPIC_API_KEY"
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_key_env_var}"
        )

    # Map parameters for Anthropic
    anthropic_params = {
        "model": params["model_name"],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "timeout": params["timeout"],
        "api_key": api_key,
    }

    # Add base_url if specified
    if "base_url" in params:
        anthropic_params["base_url"] = params["base_url"]

    # Add any additional parameters
    additional = {
        k: v
        for k, v in params.items()
        if k not in ["model_name", "temperature", "max_tokens", "timeout", "base_url"]
    }
    anthropic_params.update(additional)

    return ChatAnthropic(**anthropic_params)


def _create_openai_llm(config: LLMConfig, params: Dict[str, Any]) -> Any:
    """Create OpenAI GPT LLM instance."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError:
        raise ImportError(
            "langchain_openai package is required for OpenAI provider. "
            "Install with: pip install langchain-openai"
        )

    # Get API key from environment
    api_key_env_var = config.api_key_env_var or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_key_env_var}"
        )

    # Map parameters for OpenAI
    openai_params = {
        "model": params["model_name"],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "timeout": params["timeout"],
        "api_key": api_key,
    }

    # Add base_url if specified
    if "base_url" in params:
        openai_params["base_url"] = params["base_url"]

    # Add any additional parameters
    additional = {
        k: v
        for k, v in params.items()
        if k not in ["model_name", "temperature", "max_tokens", "timeout", "base_url"]
    }
    openai_params.update(additional)

    return ChatOpenAI(**openai_params)


def _create_google_llm(config: LLMConfig, params: Dict[str, Any]) -> Any:
    """Create Google Gemini LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except ImportError:
        raise ImportError(
            "langchain_google_genai package is required for Google provider. "
            "Install with: pip install langchain-google-genai"
        )

    # Get API key from environment
    api_key_env_var = config.api_key_env_var or "GOOGLE_API_KEY"
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(
            f"API key not found in environment variable: {api_key_env_var}"
        )

    # Map parameters for Google
    google_params = {
        "model": params["model_name"],
        "temperature": params["temperature"],
        "max_output_tokens": params["max_tokens"],  # Google uses max_output_tokens
        "timeout": params["timeout"],
        "google_api_key": api_key,
    }

    # Add any additional parameters
    additional = {
        k: v
        for k, v in params.items()
        if k not in ["model_name", "temperature", "max_tokens", "timeout", "base_url"]
    }
    google_params.update(additional)

    return ChatGoogleGenerativeAI(**google_params)


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
    """Get list of supported LLM providers.

    Returns:
        List of supported provider names
    """
    return ["anthropic", "openai", "google"]


def validate_llm_config(config: LLMConfig) -> list[str]:
    """Validate LLM configuration and return any issues found.

    Args:
        config: LLM configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check provider
    if config.provider.lower() not in get_supported_providers():
        issues.append(
            f"Unsupported provider: {config.provider}. "
            f"Supported providers: {', '.join(get_supported_providers())}"
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

    # Check retry_attempts
    if config.retry_attempts < 0:
        issues.append(
            f"retry_attempts must be non-negative, got: {config.retry_attempts}"
        )

    # Check API key environment variable exists
    api_key_env_var = config.api_key_env_var or get_default_api_key_env_var(
        config.provider
    )
    if not os.getenv(api_key_env_var):
        issues.append(f"API key not found in environment variable: {api_key_env_var}")

    return issues
