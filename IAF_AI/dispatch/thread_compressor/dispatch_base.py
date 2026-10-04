"""
Dispatch Base — Shared infrastructure for all dispatch strategies.

This file contains the common machinery that any dispatch strategy needs:
LLM calling, tool loops, response parsing, staging file management, and
status tracking. Strategy authors should NOT need to modify this file.

The strategy-specific orchestration logic lives in dispatch.py, which
imports from here.
"""

import json
import os
import sys
import time
import importlib
import importlib.util
import glob

from session_manager import (
    load_session,
    format_session_history,
)
from context.sliding_window import trim_records


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DISPATCH_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config():
    config_path = os.path.join(_DISPATCH_DIR, "dispatch_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sessions_dir():
    return os.path.join(_DISPATCH_DIR, "sessions")


def _staging_dir():
    return os.path.join(_DISPATCH_DIR, "staging")


# ---------------------------------------------------------------------------
# LLM resolution
# ---------------------------------------------------------------------------

def get_llm_caller(project_root):
    """Import and return lib.llm_client.call_llm from the project.
    Returns None if not available (allows testing with mock)."""
    try:
        lib_path = os.path.join(project_root, "lib")
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        from llm_client import call_llm
        return call_llm
    except ImportError as e:
        print(f"[dispatch] LLM client import failed: {e}")
        print(f"[dispatch] Hint: run 'pip install -r requirements.txt'")
        return None


def resolve_llm_endpoint(provider, global_config):
    """
    Resolve a provider name to (url, key) using global config.json.

    Args:
        provider:      Provider name, e.g. "openrouter"
        global_config: Parsed config.json dict

    Returns:
        (url, key) tuple. Falls back to default_provider if provider
        not found in config.
    """
    providers = global_config.get("providers", {})

    # Try the specified provider first
    if provider in providers:
        p = providers[provider]
        return p.get("url", ""), p.get("api_key") or p.get("key", "")

    # Fall back to default_provider
    default = global_config.get("default_provider", "")
    if default and default in providers:
        p = providers[default]
        return p.get("url", ""), p.get("api_key") or p.get("key", "")

    # Last resort: use first available provider
    if providers:
        first = next(iter(providers.values()))
        return first.get("url", ""), first.get("api_key") or first.get("key", "")

    return "", ""


def load_global_config(project_root):
    """Load the project-level config.json. Returns empty dict if missing."""
    global_config_path = os.path.join(project_root, "config.json")
    if os.path.exists(global_config_path):
        with open(global_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

def load_agent_tools(agent_id, project_root):
    """
    Load tools from an agent's tools/ directory.

    Scans agents/{agent_id}/tools/*_tools.py for TOOLS dicts.
    Each TOOLS dict maps tool_name -> {"handler": fn, "description": ...,
    "parameters": ...}.

    Returns:
        (tool_definitions, tool_functions) where:
        - tool_definitions: list of dicts for LLM API tools parameter
        - tool_functions:   dict mapping tool_name -> callable
        Returns ([], {}) if no tools found.
    """
    tools_dir = os.path.join(project_root, "agents", agent_id, "tools")
    if not os.path.isdir(tools_dir):
        return [], {}

    tool_files = glob.glob(os.path.join(tools_dir, "*_tools.py"))
    if not tool_files:
        return [], {}

    tool_definitions = []
    tool_functions = {}

    for filepath in tool_files:
        module_name = os.path.basename(filepath)[:-3]  # strip .py
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[dispatch] Warning: could not load {filepath}: {e}")
            continue

        tools_dict = getattr(module, "TOOLS", None)
        if not tools_dict or not isinstance(tools_dict, dict):
            continue

        for tool_name, tool_info in tools_dict.items():
            fn = tool_info.get("handler") or tool_info.get("function")
            if fn is None:
                print(f"[dispatch] Warning: tool '{tool_name}' in {filepath} has no handler")
                continue

            tool_functions[tool_name] = fn

            # Build tool definition for LLM API
            tool_definitions.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("parameters", {}),
                },
            })

    return tool_definitions, tool_functions


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------

def _extract_response_meta(response):
    """Extract _meta from call_llm response (non-destructive)."""
    if isinstance(response, dict) and "_meta" in response:
        return response.get("_meta", {})
    return {}


def call_with_tool_loop(
    messages, url, key, model,
    call_llm_fn, tool_definitions, tool_functions,
    max_tool_loops=10, tool_choice=None, exit_on_tools=None,
    temperature=None, top_p=None, reasoning=None, max_tokens=None,
    provider_routing=None,
):
    """
    Call LLM with tool loop support.

    If the agent has no tools, this is a single LLM call returning text.
    If the agent has tools, cycles between LLM and tool execution until
    the LLM returns a text response or max_tool_loops is reached.

    Args:
        messages:          Ready-to-send messages array
        url:               LLM API endpoint URL
        key:               API key
        model:             LLM model string
        call_llm_fn:       LLM call function: call_llm(url, key, model, messages, tools=None)
        tool_definitions:  Tool defs for LLM API (can be empty)
        tool_functions:    Dict of tool_name -> callable (can be empty)
        max_tool_loops:    Safety limit on tool call cycles
        tool_choice:       Force tool use (e.g. "required" for OpenAI-compatible APIs)
        temperature:       Sampling temperature (None = use model default)
        top_p:             Nucleus sampling threshold (None = use model default)
        reasoning:         Reasoning config dict (None = no reasoning control)

    Returns:
        (content, tool_history, response_meta) where:
        - content:       Final text response from LLM
        - tool_history:  List of tool call records (empty if no tools used)
        - response_meta: Dict with usage data from the last LLM call
    """
    tool_history = []

    # --- No tools: simple single call ---
    if not tool_definitions or not tool_functions:
        response = call_llm_fn(url, key, model, messages,
                               temperature=temperature, top_p=top_p,
                               reasoning=reasoning, max_tokens=max_tokens,
                               provider_routing=provider_routing)
        content = _extract_text(response)
        return content, tool_history, _extract_response_meta(response)

    # --- With tools: enter tool loop ---
    text_content = ""
    response_meta = {}
    for loop in range(max_tool_loops):
        response = call_llm_fn(
            url, key, model, messages,
            tools=tool_definitions, tool_choice=tool_choice,
            temperature=temperature, top_p=top_p,
            reasoning=reasoning, max_tokens=max_tokens,
            provider_routing=provider_routing,
        )
        response_meta = _extract_response_meta(response)

        # Parse response
        text_content, tool_calls = _parse_llm_response(response)

        # If LLM returned text (no tool calls)
        if not tool_calls:
            if exit_on_tools:
                # Game action not yet submitted — inject retry prompt and continue
                messages.append({
                    "role": "user",
                    "content": "Please submit your action using a tool; do not reply with text only.",
                })
                continue
            return text_content, tool_history, response_meta

        # Process each tool call
        game_tool_called = False
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", "")

            # Execute tool
            if tool_name in tool_functions:
                try:
                    tool_result = tool_functions[tool_name](tool_input)
                    if not isinstance(tool_result, str):
                        tool_result = json.dumps(tool_result, ensure_ascii=False)
                except Exception as e:
                    tool_result = f"Tool execution error: {e}"
            else:
                tool_result = f"Unknown tool: {tool_name}"

            # Track if a game action tool was called
            if exit_on_tools and tool_name in exit_on_tools:
                game_tool_called = True

            # Record to tool history (for private memory)
            tool_history.append({
                "tool": tool_name,
                "input": tool_input,
                "output": tool_result[:2000],  # truncate for memory
            })

            # Append assistant's tool request + tool result to messages
            # for the next LLM call
            # ★In DeepSeek thinking mode an assistant message carrying tool_calls must
            #   echo reasoning_content back, or the next turn returns 400.
            _assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input, ensure_ascii=False),
                    },
                }],
            }
            _rc = response.get("reasoning_content") if isinstance(response, dict) else None
            if _rc:
                _assistant_msg["reasoning_content"] = _rc
            messages.append(_assistant_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result,
            })

        # Exit immediately once a game action tool has been called
        if game_tool_called:
            return text_content, tool_history, response_meta

    # Max loops reached — return whatever text we have
    return text_content or "[Tool loop limit reached]", tool_history, response_meta


def _extract_text(response):
    """Extract text content from call_llm response.
    call_llm returns: {"role": "assistant", "content": "text here"}"""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content", "") or ""
    return str(response)


def _parse_llm_response(response):
    """
    Parse LLM response into text content and tool calls.

    call_llm returns one of:
      - {"role": "assistant", "content": "text"}          → pure text
      - {"role": "assistant", "tool_calls": [...]}         → tool calls
      - Plain string (from mock in testing)
      - List of content blocks (Anthropic-style)

    Returns:
        (text_content, tool_calls) where tool_calls is a list of
        {"id": ..., "name": ..., "input": {...}} dicts, empty if none.
    """
    # Plain string (mock or simple response)
    if isinstance(response, str):
        return response, []

    # List of content blocks (Anthropic-style)
    if isinstance(response, list):
        text_parts = []
        tool_calls = []
        for block in response:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
        return "\n".join(text_parts), tool_calls

    # Dict response
    if isinstance(response, dict):
        # Direct message dict from call_llm:
        # {"role": "assistant", "content": "text", "tool_calls": [...]}
        if "role" in response:
            text = response.get("content", "") or ""
            raw_tool_calls = response.get("tool_calls", [])
            if not raw_tool_calls:
                return text, []
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            return text, tool_calls

        # {"type": "text", "content": "..."}
        if response.get("type") == "text":
            return response.get("content", ""), []

        # {"type": "tool_use", ...}
        if response.get("type") == "tool_use":
            return "", [{
                "id": response.get("id", ""),
                "name": response.get("name", ""),
                "input": response.get("input", {}),
            }]

        # Anthropic API response: {"content": [...blocks...]}
        if "content" in response and isinstance(response["content"], list):
            return _parse_llm_response(response["content"])

        # OpenAI-style wrapper: {"choices": [{"message": {...}}]}
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return _parse_llm_response(msg)

    # Fallback
    return str(response), []


# ---------------------------------------------------------------------------
# Private agent memory
# ---------------------------------------------------------------------------

def write_agent_memory(agent_id, tool_history):
    """
    Write tool call history to staging/{agent_id}_memory.md.
    This file is only included in that agent's context_files,
    making it private to that agent.

    Creates/overwrites the file. If tool_history is empty, writes
    empty string (agent has no private memory this session).
    """
    staging = _staging_dir()
    os.makedirs(staging, exist_ok=True)
    memory_file = os.path.join(staging, f"{agent_id}_memory.md")

    if not tool_history:
        with open(memory_file, "w", encoding="utf-8") as f:
            f.write("")
        return

    parts = ["## Private Memory (Tool Results)\n"]
    for entry in tool_history:
        tool = entry.get("tool", "unknown")
        tool_input = entry.get("input", {})
        output = entry.get("output", "")

        input_str = json.dumps(tool_input, ensure_ascii=False, indent=None)
        parts.append(f"### {tool}: {input_str}")
        parts.append(f"{output}\n")

    with open(memory_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Staging file management
# ---------------------------------------------------------------------------

def write_staging_history(project_root, session_id):
    """Load session records, trim if needed, format as markdown, write to
    staging/session_history.md. Creates staging dir and file if missing."""
    staging = _staging_dir()
    os.makedirs(staging, exist_ok=True)
    staging_file = os.path.join(staging, "session_history.md")

    config = _load_config()
    max_tokens = config.get("max_history_tokens", 3000)
    trim_strategy = config.get("trim_strategy", None)

    records = load_session(_sessions_dir(), session_id)
    records = trim_records(records, max_tokens=max_tokens, trim_strategy=trim_strategy)
    formatted = format_session_history(records)
    with open(staging_file, "w", encoding="utf-8") as f:
        f.write(formatted)


def clear_staging():
    """Clear all staging files."""
    staging = _staging_dir()
    os.makedirs(staging, exist_ok=True)
    # Clear session history
    staging_file = os.path.join(staging, "session_history.md")
    with open(staging_file, "w", encoding="utf-8") as f:
        f.write("")
    # Clear all agent memory files
    for f in glob.glob(os.path.join(staging, "*_memory.md")):
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("")


# ---------------------------------------------------------------------------
# Active session status
# ---------------------------------------------------------------------------

def _status_file():
    """Path to active_session.json in staging/."""
    return os.path.join(_staging_dir(), "active_session.json")


def set_status(session_id, round_num=0, agent_id="", agent_name="", status="running"):
    """Write or update the active session status file."""
    os.makedirs(_staging_dir(), exist_ok=True)
    data = {
        "session_id": session_id,
        "status": status,
        "current_round": round_num,
        "current_agent_id": agent_id,
        "current_agent_name": agent_name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(_status_file(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def clear_status():
    """Remove the active session status file."""
    path = _status_file()
    if os.path.exists(path):
        os.remove(path)


def get_status():
    """
    Read current dispatch status.
    Returns dict with session info if running, or {"status": "idle"} if not.
    """
    path = _status_file()
    if not os.path.exists(path):
        return {"status": "idle"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"status": "idle"}
