"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json
import time

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# RateLimitError and connection errors may come from OpenAI or compatible APIs
try:
    from openai import RateLimitError, APIConnectionError
except ImportError:
    # Fallback for compatible APIs
    RateLimitError = Exception
    APIConnectionError = Exception


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        # --- System prompt for LLM ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                    "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                    "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                    "**IMPORTANT: When calling these tools, you do NOT need to provide the `kline_data` parameter - it will be automatically injected by the system.** "
                    "Just call the tools with their other parameters (like `period`, `fastperiod`, `slowperiod`, `signalperiod`) if needed. "
                    "For example, call `compute_rsi(period=14)` instead of `compute_rsi(kline_data={{...}}, period=14)`.\n\n"
                    f"⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. "
                    "You must interpret this data quickly and accurately.\n\n"
                    "Here is the OHLC data:\n{kline_data}.\n\n"
                    "Call necessary tools, and analyze the results.\n\n"
                    "**Important: Please respond in Chinese (中文). All analysis and explanations should be written in Chinese.**\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

        chain = prompt | llm.bind_tools(tools)
        messages = state["messages"]
        
        # Ensure at least one user message exists (required by some LLM APIs like DashScope)
        from langchain_core.messages import HumanMessage
        
        def ensure_user_message(msgs):
            """Ensure messages list contains at least one user/human message"""
            if not msgs or not any(getattr(msg, 'type', None) in ('human', 'user') for msg in msgs):
                return [HumanMessage(content="请分析提供的OHLC数据的技术指标。请用中文回答。")] + list(msgs)
            return list(msgs)

        # --- Retry wrapper for LLM invocation ---
        def invoke_with_retry(call_fn, *args, retries=3, wait_sec=8):
            """Retry LLM calls with exponential backoff for rate limits, connection errors, etc."""
            last_error = None
            for attempt in range(retries):
                try:
                    return call_fn(*args)
                except (RateLimitError, APIConnectionError) as e:
                    last_error = e
                    error_type = type(e).__name__
                    print(
                        f"{error_type} encountered, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                    )
                    if attempt < retries - 1:
                        time.sleep(wait_sec)
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    # Check if it's a connection-related error (including httpx errors)
                    error_str = str(e).lower()
                    error_module = type(e).__module__.lower()
                    
                    # Check for connection-related keywords or httpx/requests errors
                    connection_keywords = ['connection', 'network', 'disconnected', 'timeout', 'remote protocol']
                    is_connection_error = (
                        any(keyword in error_str for keyword in connection_keywords) or
                        'httpx' in error_module or
                        'requests' in error_module or
                        'RemoteProtocolError' in error_type or
                        'ConnectError' in error_type
                    )
                    
                    if is_connection_error:
                        print(
                            f"Connection error ({error_type}): {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                        )
                        if attempt < retries - 1:
                            time.sleep(wait_sec)
                    else:
                        # For other errors, re-raise immediately
                        raise
            raise RuntimeError(f"Max retries ({retries}) exceeded. Last error: {last_error}")

        # --- Step 1: Ask for tool calls ---
        messages = ensure_user_message(messages)
        ai_response = invoke_with_retry(chain.invoke, {"messages": messages})
        messages.append(ai_response)

        # --- Step 2: Collect tool results ---
        if hasattr(ai_response, "tool_calls"):
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                # Always provide kline_data
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                # Lookup tool by name
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                # Append result as ToolMessage
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # --- Step 3: Re-run the chain with tool results ---
        # Ensure user message still exists after tool calls
        messages = ensure_user_message(messages)
        final_response = invoke_with_retry(chain.invoke, {"messages": messages})

        return {
            "messages": messages + [final_response],
            "indicator_report": final_response.content,
        }

    return indicator_agent_node
