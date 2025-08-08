"""Example usage of LLM configuration system."""

from src.orchestration.config import LLMConfig
from src.orchestration.llm_factory import validate_llm_config


def example_basic_usage():
    """Basic example of using LLM configuration."""
    print("=== Basic LLM Configuration Usage ===")

    # Example 1: Using default configuration (Claude)
    default_config = LLMConfig()
    print(f"Default provider: {default_config.provider}")
    print(f"Default model: {default_config.model_name}")
    print(f"Default temperature: {default_config.temperature}")
    print()

    # Example 2: Custom OpenAI configuration
    openai_config = LLMConfig(
        provider="openai", model_name="gpt-4", temperature=0.7, max_tokens=2048
    )
    print(f"OpenAI config: {openai_config.provider}/{openai_config.model_name}")
    print()

    # Example 3: Google/Gemini configuration
    google_config = LLMConfig(
        provider="google",
        model_name="gemini-pro",
        temperature=0.5,
        api_key_env_var="GOOGLE_API_KEY",
    )
    print(f"Google config: {google_config.provider}/{google_config.model_name}")
    print()


def example_validation():
    """Example of configuration validation."""
    print("=== Configuration Validation ===")

    # Valid configuration
    valid_config = LLMConfig(provider="anthropic", temperature=0.3)
    issues = validate_llm_config(valid_config)
    print(f"Valid config issues: {issues}")

    # Invalid configuration
    invalid_config = LLMConfig(provider="unknown", temperature=1.5)
    issues = validate_llm_config(invalid_config)
    print(f"Invalid config issues: {len(issues)} found:")
    for issue in issues:
        print(f"  - {issue}")
    print()


def example_environment_variables():
    """Example of using environment variables."""
    print("=== Environment Variable Configuration ===")

    # Set some example environment variables
    env_vars = {
        "DIVERSIFIER_LLM_PROVIDER": "openai",
        "DIVERSIFIER_LLM_MODEL_NAME": "gpt-3.5-turbo",
        "DIVERSIFIER_LLM_TEMPERATURE": "0.8",
        "DIVERSIFIER_LLM_MAX_TOKENS": "1024",
    }

    print("Environment variables to set:")
    for key, value in env_vars.items():
        print(f"  export {key}={value}")
    print()

    # Note: In practice, these would be set in your shell/environment
    # The configuration system will automatically pick them up


def example_toml_config():
    """Example of TOML configuration file."""
    print("=== TOML Configuration Example ===")

    toml_example = """
# Diversifier Configuration

[llm]
# Choose provider: "anthropic", "openai", "google"
provider = "anthropic"
model_name = "claude-3-5-sonnet-20241022"
temperature = 0.1
max_tokens = 4096
timeout = 120
retry_attempts = 3

# Optional: Custom API key environment variable
# api_key_env_var = "MY_CUSTOM_API_KEY"

# Optional: Custom API endpoint
# base_url = "https://api.example.com"

# Examples for other providers:
# For OpenAI:
# provider = "openai"
# model_name = "gpt-4"
# temperature = 0.7

# For Google:
# provider = "google"  
# model_name = "gemini-pro"
# temperature = 0.5
"""

    print("Example diversifier.toml:")
    print(toml_example)


def example_factory_usage():
    """Example of using the LLM factory (requires LangChain packages)."""
    print("=== LLM Factory Usage ===")

    print("To use the LLM factory, install the required LangChain packages:")
    print("  # For Anthropic")
    print("  pip install langchain-anthropic")
    print("  # For OpenAI")
    print("  pip install langchain-openai")
    print("  # For Google")
    print("  pip install langchain-google-genai")
    print()

    print("Then set your API key:")
    print("  export ANTHROPIC_API_KEY=your_key_here")
    print("  export OPENAI_API_KEY=your_key_here")
    print("  export GOOGLE_API_KEY=your_key_here")
    print()

    print("Example factory usage:")
    print(
        """
from src.orchestration.llm_factory import create_llm_from_config
from src.orchestration.config import LLMConfig

# Create LLM from configuration
config = LLMConfig(provider="anthropic", temperature=0.5)
llm = create_llm_from_config(config)

# Use with LangChain
response = llm.invoke("Hello, world!")
print(response.content)
"""
    )


def main():
    """Run all examples."""
    example_basic_usage()
    example_validation()
    example_environment_variables()
    example_toml_config()
    example_factory_usage()


if __name__ == "__main__":
    main()
