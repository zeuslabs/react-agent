"""Define the configurable parameters for the agent."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Literal

from langchain_core.runnables import ensure_config, RunnableConfig
from langgraph.config import get_config
from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    # LLM Provider 선택 (openai 또는 bedrock)
    llm_provider: Literal["openai", "bedrock"] = field(
        default="bedrock",
        metadata={
            "description": "The LLM provider to use. Options: 'openai' or 'bedrock'"
        },
    )

    # OpenAI 모델 설정
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o",
        metadata={
            "description": "The name of the OpenAI language model to use. "
            "Examples: gpt-4o, gpt-4-turbo, gpt-3.5-turbo"
        },
    )

    # Amazon Bedrock 모델 설정
    bedrock_model_id: str = field(
        default="anthropic.claude-3-5-sonnet-20240620-v1:0",
        metadata={
            "description": "The Bedrock model ID to use. "
            "Examples: "
            "- anthropic.claude-3-5-sonnet-20240620-v1:0 (Claude 3.5 Sonnet)"
            "- anthropic.claude-3-haiku-20240307-v1:0 (Claude 3 Haiku)"
        },
    )

    # AWS Bedrock 설정
    aws_region: str = field(
        default="ap-northeast-2",
        metadata={
            "description": "AWS region for Bedrock service. "
            "Examples: us-east-1, us-west-2, ap-northeast-2, eu-west-1"
        },
    )

    aws_profile: str | None = field(
        default=None,
        metadata={
            "description": "AWS credentials profile name. "
            "If not set, uses default credentials or environment variables."
        },
    )

    # 모델 파라미터
    temperature: float = field(
        default=0.1,
        metadata={
            "description": "The temperature for model sampling. "
            "Higher values (e.g., 0.8) make output more random, "
            "lower values (e.g., 0.2) make it more deterministic."
        },
    )

    max_tokens: int = field(
        default=4096,
        metadata={
            "description": "Maximum number of tokens to generate in the response."
        },
    )

    # 검색 설정
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from the current context."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
