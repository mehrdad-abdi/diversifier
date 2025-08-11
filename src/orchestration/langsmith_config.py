"""LangSmith tracing configuration for Diversifier."""

import os
from typing import Optional


def setup_langsmith_tracing() -> bool:
    """Configure LangSmith tracing if enabled via environment variables.

    Expected environment variables:
    - LANGSMITH_TRACING: Set to "true" to enable tracing
    - LANGSMITH_ENDPOINT: LangSmith API endpoint (defaults to https://api.smith.langchain.com)
    - LANGSMITH_API_KEY: API key for LangSmith
    - LANGSMITH_PROJECT: Project name for traces

    Returns:
        True if tracing was successfully configured, False otherwise
    """
    # Check if tracing is enabled
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

    if not tracing_enabled:
        return False

    # Get required environment variables
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT", "diversifier")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key:
        print("Warning: LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set")
        return False

    # Set LangSmith environment variables for the langsmith package
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    print(f"✅ LangSmith tracing enabled for project: {project}")
    return True


def get_langsmith_status() -> dict[str, Optional[str]]:
    """Get current LangSmith configuration status.

    Returns:
        Dictionary containing LangSmith configuration information
    """
    return {
        "tracing_enabled": os.getenv("LANGSMITH_TRACING", "false"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT"),
        "project": os.getenv("LANGSMITH_PROJECT"),
        "api_key_set": "Yes" if os.getenv("LANGSMITH_API_KEY") else "No",
        "langchain_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }
