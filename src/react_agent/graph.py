"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import os
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS


def get_llm(configuration: Configuration):
    """
    Get the appropriate LLM based on configuration.

    Args:
        configuration: Configuration object containing LLM settings

    Returns:
        Configured LLM instance (ChatOpenAI or ChatBedrock)
    """
    print(configuration.llm_provider)
    if configuration.llm_provider == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(
                "langchain-aws is required for Bedrock support. "
                "Install it with: pip install langchain-aws boto3"
            )

        # Bedrock LLM 초기화
        bedrock_kwargs = {
            "model_id": configuration.bedrock_model_id,
            "region_name": configuration.aws_region,
            "model_kwargs": {
                "temperature": configuration.temperature,
                "max_tokens": configuration.max_tokens,
            },
        }

        # AWS Profile이 설정된 경우 추가
        if configuration.aws_profile:
            bedrock_kwargs["credentials_profile_name"] = configuration.aws_profile

        return ChatBedrock(**bedrock_kwargs)

    elif configuration.llm_provider == "openai":
        # OpenAI LLM 초기화
        return ChatOpenAI(
            model=configuration.model,
            temperature=configuration.temperature,
            max_tokens=configuration.max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {configuration.llm_provider}. "
            "Supported providers: 'openai', 'bedrock'"
        )


# Define the function that calls the model
async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Get the appropriate LLM based on configuration
    llm = get_llm(configuration)

    # Bind tools to the model
    model = llm.bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
