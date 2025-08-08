"""Global test configuration and fixtures."""

import os
import pytest
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with required API keys and mocks."""
    # Set dummy OpenAI API key to prevent real API calls
    os.environ["OPENAI_API_KEY"] = "test-key-12345"
    
    # Set other environment variables that might be needed
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:1984"
    
    yield
    
    # Cleanup is handled by pytest automatically


@pytest.fixture(autouse=True)  
def mock_init_chat_model():
    """Globally mock init_chat_model to prevent real API calls."""
    with patch("src.orchestration.agent.init_chat_model") as mock_init:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Mock response")
        mock_init.return_value = mock_llm
        yield mock_init


@pytest.fixture(autouse=True)
def mock_create_react_agent():
    """Globally mock create_react_agent to prevent real agent creation.""" 
    with patch("src.orchestration.agent.create_react_agent") as mock_create:
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Mock agent response"}
        mock_create.return_value = mock_agent
        yield mock_create


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root directory for testing."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    return str(project_root)