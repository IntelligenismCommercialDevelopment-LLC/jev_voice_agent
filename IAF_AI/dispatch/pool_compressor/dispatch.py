"""The dispatch execution unit for jev_voice_agent -- one instance per dispatch directory.

★The class name `PresentationDispatch` is the IAF engine's generic class name (every
  dispatch.py in the framework uses it). It is not a leftover from a slide-deck project and
  must not be renamed -- rename it and `chain.load_dispatch` can no longer find it.

Public API (shared by all three copies):
    PresentationDispatch(agent_dir)
        .new_session(agent_name=...) -> sid   open one session
        .append_to_history(role, content)     push a message into the history
        .run(agent_name=...) -> dict          run once, non-streaming (the compressors' path)
        .run_single_message(msg, agent_name=...) -> dict
                                              new_session -> append -> run -> close
        .close_session()                      reset the sid

★There is no cross-turn session memory (the single-shot pattern): a new session every turn,
  with history always exactly one turn. Memory comes from the files injected through
  `context_files`, never from the history.
"""

import json
import os
import sys
import time

# Bring dispatch directory onto sys.path so sibling imports work
_DISPATCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_DISPATCH_DIR))
if _DISPATCH_DIR not in sys.path:
    sys.path.insert(0, _DISPATCH_DIR)

from dispatch_base import (
    get_llm_caller,
    load_global_config,
    resolve_llm_endpoint,
    call_with_tool_loop,
)
import session_io

# Fix dispatch_base path resolution for THIS dispatch directory.
import dispatch_base as _dispatch_base_module
_dispatch_base_module._DISPATCH_DIR = _DISPATCH_DIR


# ---------------------------------------------------------------------------
# Context path resolution — supports "session:" prefix
# ---------------------------------------------------------------------------

def _resolve_context_path(dispatch_dir, path, sid):
    """
    Resolve a context_files entry to an absolute path.

    Convention:
      - Path with "session:" prefix -> relative to sessions/{sid}/
      - Plain path -> relative to dispatch directory
      - If sid is None and path has "session:" prefix, returns None.
    """
    if path.startswith("session:"):
        if not sid:
            return None
        rel = path[len("session:"):]
        return os.path.join(session_io.session_dir(dispatch_dir, sid), rel)
    return os.path.join(dispatch_dir, path)


def _read_context_file(dispatch_dir, path, sid):
    """Read a context file. Returns empty string if missing or empty."""
    full = _resolve_context_path(dispatch_dir, path, sid)
    if full is None:
        return ""
    try:
        with open(full, "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return ""


def _expand_ctx_dirs(dispatch_dir, context_files, sid):
    """Entries in context_files that point at a DIRECTORY are expanded into that directory's
    top-level files (sorted, non-recursive, subdirectories skipped); file entries are kept
    as they are. Returns a list of dispatch_dir-relative paths. Purely additive: only
    directory entries are affected."""
    _expanded = []
    for _p in context_files:
        try:
            _r = _resolve_context_path(dispatch_dir, _p, sid)
        except Exception:
            _r = None
        if _r and os.path.isdir(_r):
            for _n in sorted(os.listdir(_r)):
                _fp = os.path.join(_r, _n)
                if os.path.isfile(_fp):
                    _expanded.append(os.path.relpath(_fp, dispatch_dir))
        else:
            _expanded.append(_p)
    return _expanded


def _build_context(dispatch_dir, context_files, sid):
    """
    Read context_files and split:
      - Regular files -> system prompt text
      - _tools.json files -> tool definitions for LLM tools parameter

    Returns:
        (system_prompt: str, external_tools: list[dict])
    """
    system_parts = []
    external_tools = []

    for path in _expand_ctx_dirs(dispatch_dir, context_files, sid):
        content = _read_context_file(dispatch_dir, path, sid)
        if not content:
            continue

        # Route: _tools.json -> tools parameter
        resolved = _resolve_context_path(dispatch_dir, path, sid)
        if resolved and resolved.endswith("_tools.json"):
            try:
                tool_defs = json.loads(content)
                if isinstance(tool_defs, list):
                    external_tools.extend(tool_defs)
                else:
                    external_tools.append(tool_defs)
            except json.JSONDecodeError:
                pass
        else:
            _lbl = os.path.basename(_resolve_context_path(dispatch_dir, path, sid) or path)
            system_parts.append(f"# [{_lbl}]\n\n{content}")

    system = "\n\n---\n\n".join(system_parts) if system_parts else "You are a helpful assistant."
    return system, external_tools


# ---------------------------------------------------------------------------
# History loading — excludes compacted rounds
# ---------------------------------------------------------------------------

def _load_conversation_history(dispatch_dir, sid, max_turns=50):
    """Load history messages, excluding rounds covered by trimmed_marker."""
    if not sid:
        return []
    records = session_io.load_history(dispatch_dir, sid)

    compacted_rounds = set()
    for rec in records:
        if rec.get("type") == "trimmed_marker":
            for r in rec.get("compacted_rounds", []):
                compacted_rounds.add(r)

    messages = []
    round_num = 0
    i = 0
    conv_records = [r for r in records if r.get("type") in ("user_input", "agent_response")]
    while i < len(conv_records):
        rec = conv_records[i]
        if rec.get("type") == "user_input":
            round_num += 1
            if round_num in compacted_rounds:
                i += 1
                if i < len(conv_records) and conv_records[i].get("type") == "agent_response":
                    i += 1
                continue
            messages.append({"role": "user", "content": rec.get("content", "")})
            if i + 1 < len(conv_records) and conv_records[i + 1].get("type") == "agent_response":
                messages.append({"role": "assistant", "content": conv_records[i + 1].get("content", "")})
                i += 2
            else:
                i += 1
        else:
            i += 1

    return messages[-max_turns * 2:]


# ---------------------------------------------------------------------------
# Tool loading — local tools from tools/*.py
# ---------------------------------------------------------------------------

def _load_dispatch_tools(dispatch_dir):
    """Load tools from dispatch's tools/*_tools.py directory."""
    import glob
    import importlib.util

    tools_dir = os.path.join(dispatch_dir, "tools")
    if not os.path.isdir(tools_dir):
        return [], {}

    tool_definitions = []
    tool_functions = {}

    for filepath in sorted(glob.glob(os.path.join(tools_dir, "*_tools.py"))):
        module_name = os.path.basename(filepath)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[dispatch] Could not load {filepath}: {e}")
            continue

        tools_dict = getattr(module, "TOOLS", None)
        if not tools_dict:
            continue

        for tool_name, tool_info in tools_dict.items():
            fn = tool_info.get("handler") or tool_info.get("function")
            if fn is None:
                continue
            tool_functions[tool_name] = fn
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
# External tool handler -- a fallback for tools declared by a `_tools.json` in context_files
# ★None of jev_voice_agent's three configs has a `_tools.json`, so this path is always empty
#   (external_tools=[] in a real run). Reported as such and deliberately kept.
# ---------------------------------------------------------------------------

def _make_external_tool_handler(tool_name):
    """Build an acknowledge-only, never-execute handler for an externally declared tool; the
    layer above interprets it itself."""
    def handler(args):
        return json.dumps(
            {"status": "accepted", "tool": tool_name, "input": args},
            ensure_ascii=False,
        )
    return handler


# ---------------------------------------------------------------------------
# Reasoning data saving
# ---------------------------------------------------------------------------

_reasoning_file_cache = {}


def _get_reasoning_filepath(output_dir, step_name):
    """Reuse one filename per step per batch, so several agents append their records to the
    same file."""
    if step_name not in _reasoning_file_cache:
        ts = time.strftime("%y_%m_%d_%H%M%S")
        _reasoning_file_cache[step_name] = os.path.join(
            output_dir, f"{step_name}_{ts}.json"
        )
    return _reasoning_file_cache[step_name]


def _save_reasoning_if_configured(response_meta, reasoning_config,
                                   reasoning_output_path, agent_dir,
                                   agent_conf, config):
    """Save reasoning output to a JSON file (append mode).
    Skipped when reasoning_output_path is None."""
    if not reasoning_output_path:
        return

    usage = response_meta.get("usage", {})

    # Derive the step name from agent_dir: .../<step>/dispatch/<name>/ -> "<step>"
    step_name = os.path.basename(os.path.dirname(os.path.dirname(agent_dir)))

    # Parse reasoning_tokens (tolerating different response shapes)
    details = usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens", 0)

    record = {
        "step": step_name,
        "agent_id": config.get("agent_id", ""),
        "module": "script_writer",
        "model": agent_conf.get("model", ""),
        "reasoning_effort": (reasoning_config or {}).get("effort", ""),
        "reasoning_text": "",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": reasoning_tokens,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    filepath = _get_reasoning_filepath(reasoning_output_path, step_name)

    # append to the JSON array
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    existing.append(record)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PresentationDispatch
# ---------------------------------------------------------------------------

class PresentationDispatch:
    """The execution unit for one dispatch directory. Loading is `chain.load_dispatch()`'s job.

    Usage (non-streaming, the compressors' path):
        disp = chain.load_dispatch("thread_compressor")
        out = disp.run_single_message("The material to compress is in compress_input.md", agent_name="thread_compressor")
        print(out["text_response"])

    Usage (streaming, the speaker's path):
        disp.new_session(agent_name="core")
        disp.append_to_history("user", user_text)
        async for ev in disp.run_stream(agent_name="core"):
            ...
        disp.close_session()
    """

    def __init__(self, agent_dir: str):
        """
        Initialize dispatch instance.

        Args:
            agent_dir: the dispatch directory, e.g. 'IAF_AI/dispatch/speaker'
        """
        self.agent_dir = os.path.abspath(agent_dir)
        self.project_root = os.path.dirname(os.path.dirname(self.agent_dir))
        self.config = self._load_config()
        self.session_id = None

        # Fix path resolution for all sibling modules
        self._fix_sibling_paths()

    def _fix_sibling_paths(self):
        """Override _DISPATCH_DIR in all cached sibling modules.

        When multiple PresentationDispatch instances coexist, Python's module cache
        causes all agents to share the same sibling modules (session_io, etc.).
        This method patches _DISPATCH_DIR so each agent writes to its own directory.
        """
        _dispatch_base_module._DISPATCH_DIR = self.agent_dir
        for mod_name in ('session_io',):
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, '_DISPATCH_DIR'):
                mod._DISPATCH_DIR = self.agent_dir
            if mod and hasattr(mod, '_PROJECT_ROOT'):
                mod._PROJECT_ROOT = self.project_root

    def _load_config(self):
        config_path = os.path.join(self.agent_dir, "dispatch_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def new_session(self, session_id: str = None, agent_name: str = "core_chat"):
        """
        Open one session (under the single-shot pattern this is one conversational turn).

        Returns:
            str: The session_id
        """
        self._fix_sibling_paths()
        agent_conf = self.config["agents"][agent_name]
        model = agent_conf.get("model", "")
        provider = agent_conf.get("provider", "")
        self.session_id = session_io.new_session(
            self.agent_dir, model=model, provider=provider,
        )
        return self.session_id

    def run(self, retry: bool = False, agent_name: str = "core_chat") -> dict:
        """
        Execute one dispatch call.

        1. Read config's context_files (split: text -> system, _tools.json -> tools)
        2. Assemble prompt (context_files content + conversation history)
        3. Call LLM with tool loop
        4. Return tool call results

        Args:
            retry: If True, append a retry hint to history before calling LLM

        Returns:
            dict: {
                "tool_calls": [],        # ★always empty here: none of the three configs loads tools
                "text_response": "...",
                "duration_ms": N,
            }
        """
        if not self.session_id:
            raise RuntimeError("No active session. Call new_session() first.")

        self._fix_sibling_paths()
        config = self.config
        sid = self.session_id

        # --- Resolve LLM caller ---
        call_llm_fn = get_llm_caller(self.project_root)
        if call_llm_fn is None:
            raise RuntimeError("No LLM caller available")

        # --- Resolve endpoint ---
        global_config = load_global_config(self.project_root)
        agent_conf = config["agents"][agent_name]
        provider = agent_conf.get("provider", "anthropic")
        model = agent_conf.get("model", "")
        temperature = agent_conf.get("temperature")
        top_p = agent_conf.get("top_p")
        reasoning = agent_conf.get("reasoning")
        reasoning_output_path = agent_conf.get("reasoning_data_output_path")
        max_tokens = agent_conf.get("max_tokens")
        url, key = resolve_llm_endpoint(provider, global_config)

        # --- Retry hint ---
        if retry:
            session_io.append_history(self.agent_dir, sid, {
                "type": "user_input",
                "content": "Your previous response did not include the required tool call. "
                           "Please use the available tools to submit your action.",
            })

        # --- Build context: system prompt + external tool definitions ---
        context_files = agent_conf.get("context_files", [])
        system_prompt, external_tools = _build_context(
            self.agent_dir, context_files, sid,
        )

        history = _load_conversation_history(self.agent_dir, sid)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        # --- Load tools (mode-dependent) ---
        free_search = agent_conf.get("free_search")

        if free_search and (free_search.get("free_search_path") or free_search.get("write_allow")):
            # ═══ FREE SEARCH MODE ═══
            # Load explore_tools.py with path fencing via factory
            import importlib.util as _ilu
            explore_path = os.path.join(self.agent_dir, "tools", "explore_tools.py")
            _spec = _ilu.spec_from_file_location("explore_tools", explore_path)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)

            read_roots = [os.path.realpath(p)
                          for p in free_search["free_search_path"]]
            write_root = os.path.realpath(free_search.get("output_path", ""))
            if write_root:
                os.makedirs(write_root, exist_ok=True)

            explore_dict = _mod.make_tools(read_roots, write_root)
            tool_definitions = []
            tool_functions = {}
            for tname, tinfo in explore_dict.items():
                tool_functions[tname] = tinfo["handler"]
                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": tname,
                        "description": tinfo["description"],
                        "parameters": tinfo["parameters"],
                    },
                })

            tc = free_search.get("tool_choice", "auto")
            max_loops = free_search.get("max_tool_loops", 40)
            exit_tools = {"write_file"} if free_search.get("exit_on_write", True) else None

        else:
            # ═══ STANDARD MODE (TCR / chat / interview) ═══
            tool_definitions, tool_functions = _load_dispatch_tools(self.agent_dir)

            # Register external tools with acknowledgment handlers
            for ext_tool in external_tools:
                func_info = ext_tool.get("function", {})
                name = func_info.get("name", "")
                if name and name not in tool_functions:
                    tool_functions[name] = _make_external_tool_handler(name)
                    tool_definitions.append(ext_tool)

            # Determine tool_choice based on provider
            # Only force tool use when externally declared tools exist (never the case here)
            tc = None
            if external_tools and tool_definitions:
                if provider in ("openrouter", "openai", "xai"):
                    tc = "required"
                elif provider == "anthropic":
                    tc = {"type": "any"}

            exit_tools = ({t["function"]["name"] for t in external_tools}
                          if external_tools else None)
            max_loops = 10

            # -- Live STATUS trail: in STANDARD MODE custom tools emit no events by
            #    default, so wrap the handler one layer deeper (same approach as
            #    explore_tools, best-effort, degrading silently without status_log) to make
            #    the tool loop write STATUS per tool in real time.
            #    Built into the template, so a new organ copied from it and configured in
            #    STANDARD MODE gets the live per-event STATUS for free.
            try:
                import time as _sl_time
                import status_log as _sl
            except Exception:
                _sl = None
            if _sl is not None:
                def _wrap_sl(_h, _nm):
                    def _inner(inp):
                        _sl.tool_start(_nm, inp)
                        _t0 = _sl_time.time()
                        try:
                            out = _h(inp)
                        except Exception as e:
                            _sl.tool_done(_nm, _sl_time.time() - _t0, err=e)
                            raise
                        _sl.tool_done(_nm, _sl_time.time() - _t0, out=out)
                        return out
                    return _inner
                tool_functions = {_n: _wrap_sl(_h, _n) for _n, _h in tool_functions.items()}

        session_io.log_call(self.agent_dir, sid,
                            "call_started", model=model, provider=provider)

        t0 = time.time()
        tool_history = []
        response_meta = {}
        content, tool_history, response_meta = call_with_tool_loop(
            messages=messages,
            url=url, key=key, model=model,
            call_llm_fn=call_llm_fn,
            tool_definitions=tool_definitions,
            tool_functions=tool_functions,
            max_tool_loops=max_loops,
            tool_choice=tc,
            exit_on_tools=exit_tools,
            temperature=temperature,
            top_p=top_p,
            reasoning=reasoning,
            max_tokens=max_tokens,
            provider_routing=agent_conf.get("provider_routing"),
        )

        duration_ms = int((time.time() - t0) * 1000)

        session_io.log_call(self.agent_dir, sid, "call_completed",
                            duration_ms=duration_ms,
                            reply_length=len(content),
                            tool_count=len(tool_history),
                            # ★Write usage to disk. It used to be returned by
                            #   call_with_tool_loop and never written -- which made the cost
                            #   of compression completely invisible, and compression is the
                            #   main source of background cost. All three copies of
                            #   dispatch.py must agree here.
                            usage=(response_meta or {}).get("usage", {}))

        # Record agent response in history
        session_io.append_history(self.agent_dir, sid, {
            "type": "agent_response",
            "agent_id": config.get("agent_id", "agent_001"),
            "display_name": agent_conf.get("display_name", "Agent"),
            "round": 1,
            "content": content,
            "tool_count": len(tool_history),
        })

        # --- Save reasoning data if configured ---
        _save_reasoning_if_configured(
            response_meta=response_meta,
            reasoning_config=reasoning,
            reasoning_output_path=reasoning_output_path,
            agent_dir=self.agent_dir,
            agent_conf=agent_conf,
            config=config,
        )

        # --- Build structured result ---
        # Report only externally declared tools upwards; internal file tools are filtered
        # out (tool_history is always empty here).
        tool_calls = []
        for th in tool_history:
            if not exit_tools or th["tool"] in exit_tools:
                tool_calls.append({
                    "name": th["tool"],
                    "input": th["input"],
                })

        return {
            "tool_calls": tool_calls,
            "text_response": content,
            "duration_ms": duration_ms,
        }

    def append_to_history(self, role: str, content: str):
        """
        Append a message to the history. Use it to push user input before run()/run_stream().

        Args:
            role: "user" or "assistant"
            content: Message content
        """
        if not self.session_id:
            raise RuntimeError("No active session.")

        self._fix_sibling_paths()
        if role == "user":
            session_io.append_history(self.agent_dir, self.session_id, {
                "type": "user_input",
                "content": content,
            })
        else:
            session_io.append_history(self.agent_dir, self.session_id, {
                "type": "agent_response",
                "agent_id": self.config.get("agent_id", "agent_001"),
                "content": content,
            })

    def close_session(self):
        """
        Reset the session.

        ★This method must not be deleted: run_single_message() calls it at the end, and
        self.session_id = None is exactly what lets run_role be called repeatedly (only then
        does the next new_session really create one).
        There used to be a lazily loaded session_close here doing persona-memory updates;
        session_close.enabled was always false, and those 313 lines have been deleted.
        """
        if not self.session_id:
            return
        self.session_id = None

    # ------------------------------------------------------------------
    # convenience entry point
    # ------------------------------------------------------------------

    def run_single_message(self, user_message: str, agent_name: str = "core_chat") -> dict:
        """
        Single-turn LLM call: new_session → inject user message → run → close.

        Args:
            user_message: The user prompt
            agent_name:   Which agent config to use (default: "core_chat")

        Returns:
            dict with "text_response", "tool_calls", "duration_ms"
        """
        self.new_session(agent_name=agent_name)
        self.append_to_history("user", user_message)
        result = self.run(agent_name=agent_name)
        self.close_session()
        return result
