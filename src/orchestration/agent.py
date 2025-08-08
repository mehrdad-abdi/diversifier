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


class AgentType(Enum):
    """Types of agents used in the diversification process."""

    ANALYZER = "analyzer"
    MIGRATOR = "migrator"
    TESTER = "tester"
    REPAIRER = "repairer"


class DiversificationAgent:
    """LangChain agent for coordinating diversification tasks."""

    def __init__(
        self,
        agent_type: AgentType,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the LangChain agent.

        Args:
            agent_type: Type of agent (analyzer, migrator, tester, repairer)
            model_name: OpenAI model to use
            temperature: Temperature for LLM responses
            max_tokens: Maximum tokens for responses
            tools: List of tools available to the agent
        """
        self.agent_type = agent_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
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
            self.llm = init_chat_model(
                f"openai:{self.model_name}", temperature=self.temperature
            )
            self.logger.info(
                f"Initialized {self.model_name} LLM for {self.agent_type.value} agent"
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

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        """Initialize the agent manager.

        Args:
            model_name: OpenAI model to use for all agents
            temperature: Temperature for all agents
        """
        self.model_name = model_name
        self.temperature = temperature
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
                model_name=self.model_name,
                temperature=self.temperature,
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
