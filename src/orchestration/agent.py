"""LangChain agent initialization and configuration."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import LLMConfig, get_config


class AgentType(Enum):
    """Types of agents used in the diversification process."""

    ANALYZER = "analyzer"
    MIGRATOR = "migrator"
    TESTER = "tester"
    REPAIRER = "repairer"
    DOC_ANALYZER = "doc_analyzer"
    SOURCE_CODE_ANALYZER = "source_code_analyzer"
    ACCEPTANCE_TEST_GENERATOR = "acceptance_test_generator"


class DiversificationAgent:
    """LangChain agent for coordinating diversification tasks."""

    def __init__(
        self,
        agent_type: AgentType,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the LangChain agent.

        Args:
            agent_type: Type of agent (analyzer, migrator, tester, repairer)
            llm_config: LLM configuration to use. If None, uses global config.
            tools: List of tools available to the agent
        """
        self.agent_type = agent_type
        self.llm_config = llm_config or get_config().llm
        self.tools = tools or []
        self.agent_executor: Any = None

        self.logger = logging.getLogger(f"diversifier.agent.{agent_type.value}")

        # Initialize LLM and memory
        self._initialize_llm()
        self.memory = MemorySaver()

        # Initialize agent
        self._initialize_agent()

    def _initialize_llm(self) -> None:
        """Initialize the LLM with configuration."""
        try:
            # Map provider names to init_chat_model format
            provider_map = {
                "anthropic": "anthropic",
                "openai": "openai",
                "google": "google_genai",  # Use google_genai for LangChain Google GenAI
            }

            provider = provider_map.get(self.llm_config.provider.lower())
            if not provider:
                # Fall back to original provider name if not in map
                provider = self.llm_config.provider.lower()

            # Create the model identifier for init_chat_model
            model_id = f"{provider}:{self.llm_config.model_name}"

            # Initialize the LLM using init_chat_model with configuration
            # Note: Only pass parameters that are commonly supported across all providers
            init_params = {
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
            }

            # Add additional params but filter out potentially unsupported ones
            for key, value in self.llm_config.additional_params.items():
                if key not in ["timeout"]:  # Skip timeout as it may not be supported
                    init_params[key] = value  # type: ignore[assignment]

            self.llm = init_chat_model(model=model_id, **init_params)

            self.logger.info(
                f"Initialized {self.llm_config.provider}:{self.llm_config.model_name} "
                f"LLM for {self.agent_type.value} agent"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with tools and prompt."""
        try:
            # Create agent
            if self.tools:
                self.agent_executor = create_react_agent(
                    self.llm, self.tools, checkpointer=self.memory
                )
            else:
                # For agents without tools, we'll use the LLM directly
                self.agent_executor = None

            self.logger.info(
                f"Initialized {self.agent_type.value} agent with {len(self.tools)} tools"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise

    def _load_agent_prompt(self) -> str:
        """Load the appropriate prompt template for this agent type from file."""
        prompt_dir = Path(__file__).parent / "prompts"

        # Map agent types to prompt files
        prompt_files = {
            AgentType.ANALYZER: "analyzer.txt",
            AgentType.MIGRATOR: "migrator.txt",
            AgentType.TESTER: "tester.txt",
            AgentType.REPAIRER: "repairer.txt",
            AgentType.DOC_ANALYZER: "doc_analyzer.txt",
            AgentType.SOURCE_CODE_ANALYZER: "source_code_analyzer.txt",
            AgentType.ACCEPTANCE_TEST_GENERATOR: "acceptance_test_generator.txt",
        }

        prompt_file = prompt_files.get(self.agent_type)
        if prompt_file:
            prompt_path = prompt_dir / prompt_file
            if prompt_path.exists():
                return prompt_path.read_text().strip()

        # Fallback for unknown agent types
        return "You are a helpful assistant for Python code migration tasks."

    def _get_agent_prompt(self) -> ChatPromptTemplate:
        """Get the appropriate prompt template for this agent type."""
        system_message = self._load_agent_prompt()

        if self.tools:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_message),
                    ("user", "{input}"),
                    (
                        "assistant",
                        "I'll help you with that task. Let me use the available tools to gather information and provide assistance.",
                    ),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_message), ("user", "{input}")]
            )

        return prompt

    def invoke(self, input_text: str) -> Dict[str, Any]:
        """Invoke the agent with input text.

        Args:
            input_text: Input text for the agent

        Returns:
            Agent response dictionary
        """
        try:
            if self.agent_executor:
                config = {
                    "configurable": {"thread_id": f"agent_{self.agent_type.value}"}
                }
                result = self.agent_executor.invoke(
                    {"messages": [{"role": "user", "content": input_text}]}, config
                )
            else:
                # For agents without tools, use LLM directly
                prompt = self._get_agent_prompt()
                messages = prompt.format_messages(input=input_text)
                response = self.llm.invoke(messages)
                result = {"output": response.content}

            self.logger.info(
                f"Agent {self.agent_type.value} completed task successfully"
            )
            return result

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent.

        Args:
            tool: Tool to add to the agent
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.logger.info(f"Added tool {tool.name} to {self.agent_type.value} agent")

            # Reinitialize agent with new tools
            self._initialize_agent()

    def clear_memory(self) -> None:
        """Clear the agent's conversation memory."""
        # MemorySaver doesn't have a direct clear method, so we reinitialize it
        self.memory = MemorySaver()
        self.logger.info(f"Cleared memory for {self.agent_type.value} agent")


class AgentManager:
    """Manager for coordinating multiple diversification agents."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize the agent manager.

        Args:
            llm_config: LLM configuration to use for all agents. If None, uses global config.
        """
        self.llm_config = llm_config or get_config().llm
        self.agents: Dict[AgentType, DiversificationAgent] = {}

        self.logger = logging.getLogger("diversifier.agent_manager")

    def get_agent(
        self, agent_type: AgentType, tools: Optional[List[BaseTool]] = None
    ) -> DiversificationAgent:
        """Get or create an agent of the specified type.

        Args:
            agent_type: Type of agent to get
            tools: Optional tools to add to the agent

        Returns:
            Diversification agent instance
        """
        if agent_type not in self.agents:
            self.agents[agent_type] = DiversificationAgent(
                agent_type=agent_type,
                llm_config=self.llm_config,
                tools=tools,
            )
            self.logger.info(f"Created new {agent_type.value} agent")

        elif tools:
            # Add any new tools to existing agent
            for tool in tools:
                self.agents[agent_type].add_tool(tool)

        return self.agents[agent_type]

    def clear_all_memories(self) -> None:
        """Clear memory for all agents."""
        for agent in self.agents.values():
            agent.clear_memory()
        self.logger.info("Cleared memory for all agents")

    def get_available_agents(self) -> List[AgentType]:
        """Get list of available agent types.

        Returns:
            List of available agent types
        """
        return list(self.agents.keys())
