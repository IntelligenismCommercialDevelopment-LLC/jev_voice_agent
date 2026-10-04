"""
chain.py -- the orchestration entry point for jev_voice_agent.

There is exactly one thing to understand about the multi-dispatch loading skeleton, and it
is a prerequisite for everything downstream: **each dispatch is loaded independently with
load_dispatch()**, which clears every cached module under IAF_AI/dispatch/ first. Without
that step the second dispatch reuses the first one's session_io / dispatch_base cache, and
both dispatches write into the same sessions/ directory -- with no error.

The responsibility boundary, which is not negotiable: deterministic orchestration and all
disk writes live in this file; the LLM only generates text.

The flow (single-shot pattern -- every call opens a new session, history is always exactly
one turn, and memory comes from the pool files injected through context_files rather than
from session history):
  1. sync_config.sync()      config_base -> engine (the mirror rule's downward direction)
  2. point_current()         workspace/current -> this run's session folder (where the
                             STATUS trail lands)
  3. run_role()              load_dispatch + run_single_message + status_log instrumentation
  4. output lands in workspace/{dispatch}/

Usage:
  python3 chain.py --dispatch speaker --agent core --message "hello"
"""

import os
import sys
import time
import asyncio
import importlib.util
import shutil

def _find_shell_root(start):
    """Walk up to the shell root = the level that holds both `IAF_AI/` and `config_base/`.

    ★It does not count dirname levels: scripts get moved, and counting levels breaks the
      moment they do -- in a way that points the path at somewhere that does not exist, so
      most call sites silently get empty content.
    ★It does not go by name either: a cluster copy must never depend on the root's name or
      location.
    ★These five lines appear in several files on purpose: boilerplate like "find the root /
      lock / atomic write" is better duplicated than hoisted into a shared script.
    """
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "IAF_AI"))
                and os.path.isdir(os.path.join(d, "config_base"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the shell root (no directory above holds both "
                               "IAF_AI/ and config_base/): " + str(start))
        d = p


IAF_AI_ROOT = os.path.dirname(os.path.abspath(__file__))   # chain.py lives in IAF_AI/
SHELL_ROOT = _find_shell_root(IAF_AI_ROOT)
WORKSPACE   = os.path.join(SHELL_ROOT, "workspace")        # root of the record folders, {session_id}/
CURRENT_LINK = os.path.join(WORKSPACE, "current")          # ★symlink -> this run's workspace/{session_id}/


def _find_cluster_root(start=SHELL_ROOT):
    """Walk up from start to the directory holding config_base/cluster_manifest.json = the
    cluster root (the agent root).
    ★The anchoring convention: an organ always lives at <cluster_root>/<group>/<unit>_iaf/.
    Anchoring on a marker file is name-free and depth-free, so it survives renaming the
    cluster, nesting it under a group directory, or running several agents.
    Use this helper to find the root when reading cluster-level shared material; never count
    dirname levels.

    ★Note: jev_voice_agent is **not inside any cluster** (walking up to the filesystem root finds
    no cluster_manifest.json), so this function raises the moment it is called. It currently
    has **zero callers**; it is kept so that joining a cluster later needs no rewrite.
    (Reported as dead code and deliberately kept.)"""
    d = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(d, "config_base", "cluster_manifest.json")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("cannot find the cluster root (no ancestor holds config_base/cluster_manifest.json): " + str(start))
        d = p


# sync_config lives at the shell root; assemble_pool / reply_list live in IAF_AI/, beside this file
for _p in (SHELL_ROOT, IAF_AI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sync_config
import assemble_pool

# A live, per-event STATUS trail (sharing the canonical tools/_shared/status_log.py with the
# engine's explore_tools). If it is missing, degrade silently -- everything still works, just
# without a trail. Written to <shell>/workspace/current/STATUS.md (the current symlink points
# at this run's session folder).
sys.path.insert(0, os.path.join(SHELL_ROOT, "tools", "_shared"))
try:
    import status_log
except Exception:
    status_log = None


def point_current(session_dir):
    """Point the workspace/current symlink at this run's {session_id} directory, replacing
    the old link. The STATUS trail uses the static path workspace/current/STATUS.md and
    physically lands in {session_id}/."""
    os.makedirs(session_dir, exist_ok=True)
    # Remove the old current by type: remove() for a symlink or file, rmtree() for a real
    # directory (on macOS, removing a directory raises PermissionError, not IsADirectoryError).
    if os.path.islink(CURRENT_LINK) or os.path.isfile(CURRENT_LINK):
        os.remove(CURRENT_LINK)
    elif os.path.isdir(CURRENT_LINK):
        shutil.rmtree(CURRENT_LINK)
    # ★A relative symlink (current -> a sibling session_id) survives moving the whole tree.
    #   An absolute path breaks once anything moves.
    os.symlink(os.path.basename(os.path.normpath(session_dir)), CURRENT_LINK)


# -- engine loading (with sibling-module eviction) ----------------------------------
def load_dispatch(dispatch_name):
    """Load one dispatch and return its PresentationDispatch instance.

    ★The for loop in the middle is the entire reason this function exists. The session_io /
    dispatch_base / session_manager modules inside a dispatch directory are imported by
    **short name** (`import session_io`), so their keys in sys.modules collide across
    dispatches: loading the second dispatch makes Python reuse the first one's module
    objects, whose _DISPATCH_DIR still points at the first dispatch's directory.
    The symptom is both dispatches writing their sessions into the same sessions/ directory,
    **with no error**.
    Clearing the cache makes each dispatch re-exec its own sibling modules, so each
    _DISPATCH_DIR is correct.
    """
    agent_dir = os.path.join(SHELL_ROOT, "IAF_AI", "dispatch", dispatch_name)
    if not os.path.isdir(agent_dir):
        raise FileNotFoundError(f"dispatch directory does not exist: {agent_dir}")
    for m in list(sys.modules):
        f = getattr(sys.modules.get(m), "__file__", "") or ""
        if os.path.join("IAF_AI", "dispatch") + os.sep in f:
            del sys.modules[m]
    if agent_dir in sys.path:
        sys.path.remove(agent_dir)
    sys.path.insert(0, agent_dir)
    module_path = os.path.join(agent_dir, "dispatch.py")
    spec = importlib.util.spec_from_file_location("dispatch", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PresentationDispatch(agent_dir)


def _role_model(dispatch_name):
    """Read model from config_base/{dispatch}/dispatch_config.json, purely for the STATUS
    trail. It reads config_base rather than the engine side because config_base is
    authoritative (the mirror rule). Returns an empty string on failure; never raises."""
    import json
    try:
        with open(os.path.join(SHELL_ROOT, "config_base", dispatch_name, "dispatch_config.json"),
                  encoding="utf-8") as f:
            cfg = json.load(f)
        for _k, a in (cfg.get("agents") or {}).items():
            if a.get("model"):
                return a["model"]
    except Exception:
        pass
    return ""


def run_role(dispatch_name, agent_name, user_msg):
    """Load a role dispatch and send it a single message. Returns the engine's structured
    result {text_response, tool_calls, duration_ms, ...}.

    run_single_message is used instead of a manual new_session + run because it calls
    close_session() itself, resetting session_id -- so the same dispatch can be called
    repeatedly within one process.
    """
    if status_log:
        status_log.llm_started(dispatch_name, _role_model(dispatch_name))
    disp = load_dispatch(dispatch_name)
    out = disp.run_single_message(user_msg, agent_name=agent_name)
    if status_log:
        status_log.response(dispatch_name, out.get("text_response", "") or "")
    return out


# -- compression (after a conversational turn, off the critical path) ---------------
def _stage_input_hint():
    """A thin trigger for the compressor. ★Keep task messages thin: the rules about output
    format live in compress_*_skill.md, and only a neutral pointer belongs here -- otherwise
    the same thing is written in two places and the two drift apart."""
    return "The material to compress is in `compress_input.md`, already in your context."


def maybe_compress(session_dir, plan=None):
    """Fire the two compressions according to their token thresholds. Returns a record of
    what ran this time (an empty `ran` when nothing triggered).

    ★What gets compressed is the pool files, not the session history -- under the
    single-shot pattern history is always one turn, so compressing it is meaningless.
    ★Call site: **after** a conversational turn. The compressors are non-streaming and must
    not sit on the critical path of a spoken reply.
    ★If a `plan` is passed in it is used as-is and no thresholds are computed --
      `voice_cli`'s `/compact` comes through here with `assemble_pool.forced_plan()`.
      **There is only one execution path**, shared by manual and automatic compression, so
      the two cannot grow different behaviour in filing or archiving.
    """
    plan = plan if plan is not None else assemble_pool.compression_plan()
    record = {"plan": plan, "ran": []}

    for peer in plan.get("threads", []):
        # ★n_older = the number of blocks handed to the compressor this time, and it must
        #   be passed straight through to place() -- recomputing the boundary loses blocks
        #   under concurrency (see assemble_pool.stage_thread_input).
        n_older = assemble_pool.stage_thread_input(session_dir, peer)
        if status_log:
            status_log.turn_header("thread_compressor", f"compressing the conversation with {peer}: {n_older} older blocks -> a faithful summary")
        out = run_role("thread_compressor", "thread_compressor", _stage_input_hint())
        text = out.get("text_response", "") or ""
        if not text.strip():
            # An empty output must never be filed: place_thread_compression would use it to
            # replace the real older conversation.
            print(f"[chain] ⚠ thread_compressor returned empty text for {peer}; skipping the "
                  f"write (the old blocks are left untouched)")
            record["ran"].append({"thread": peer, "skipped": "empty_response"})
            continue
        assemble_pool.place_thread_compression(peer, text, n_older)
        record["ran"].append({"thread": peer, "blocks": n_older, "chars": len(text)})

    if plan.get("pool"):
        assemble_pool.stage_pool_input(session_dir)
        if status_log:
            status_log.turn_header("pool_compressor", "blended compression of [semantic + tails] -> a de-personalised narrative")
        out = run_role("pool_compressor", "pool_compressor", _stage_input_hint())
        text = out.get("text_response", "") or ""
        if not text.strip():
            print("[chain] ⚠ pool_compressor returned empty text; skipping the write (semantic.md is left untouched)")
            record["ran"].append({"pool": True, "skipped": "empty_response"})
        else:
            assemble_pool.place_pool_compression(text)
            record["ran"].append({"pool": True, "chars": len(text)})

    return record


# -- a voice session (the reservoir driver + post-turn compression) ----------------
def new_voice_session(peer="user", sink=None, flush_kwargs=None):
    """Create a voice session: returns (driver, session_dir).

    ★This is the assembly point: `maybe_compress` is attached **after each driver turn
      completes** (driver.on_turn_end), off the critical path -- the compressors are
      non-streaming, and putting one in front of a spoken reply adds seconds directly to
      time to first sound.

    sink is the seam for TTS. Omit it and reply_list.console_sink is used (it prints and
    claims everything played). Plugging in real TTS is a sink swap; neither the driver nor
    this function changes.
    """
    import reply_list

    sync_config.sync()
    ws_session = f"session_{time.strftime('%Y%m%d-%H%M%S')}"
    session_dir = os.path.join(WORKSPACE, ws_session)
    point_current(session_dir)
    if status_log:
        status_log.cycle_header(ws_session, f"voice/{peer}", 1)

    disp = load_dispatch("speaker")
    drv = reply_list.VoiceDriver(
        disp, session_dir, peer=peer,
        sink=sink or reply_list.console_sink,
        flush_kwargs=flush_kwargs,
    )

    # Compress after the turn. ★What is compressed is the pool files, not the session
    # history (single-shot pattern).
    async def _after_turn(rec):
        """★★This must run on a **worker thread** (2026-09-27).

        `maybe_compress` -> `run_role` -> `run_single_message` is a **blocking** LLM call,
        and this hook is invoked from inside the `VoiceDriver.pump` coroutine -- running it
        directly **stalls the event loop**: for those few seconds mic frames pile into the
        queue with nobody consuming them, the STT receive side stops turning, and the
        silence ticker does not advance.
        This stayed hidden while the trigger was 50000, which voice never reaches (~640
        turns); lowering it to 8000 makes compression really happen about every 100 turns,
        which turns the hole from latent into certain.
        ★`reply_list.pump` awaits this coroutine (an awaitable return value is awaited), so
          the order "this turn has finished speaking -> compress -> record into rec['post']"
          is unchanged; it just no longer holds the loop.
        """
        if status_log:
            status_log.response(peer, rec.get("spoken", ""))
        return await asyncio.to_thread(maybe_compress, session_dir)

    drv.on_turn_end = _after_turn
    return drv, session_dir


def run(dispatch_name: str = "speaker",
        user_message: str = "",
        agent_name: str = "core",
        session_id: str | None = None):
    """
    Run one IAF dispatch call.

    Args:
        dispatch_name: the dispatch name (matching IAF_AI/dispatch/{name}/)
        user_message: the user input message
        agent_name: which agent to call
        session_id: ★currently has no effect. The engine's new_session() declares this
                    parameter but never uses it, and session_io.new_session() does not take
                    it at all (it generates its own sid). Passing it prints a warning rather
                    than dropping it silently.
    Returns:
        dict: {text_response, tool_calls, duration_ms}
    """
    if session_id:
        print(f"[chain] ⚠ --session {session_id} ignored: the engine does not support "
              f"choosing a session id (session_io.new_session generates its own).")

    # --- Step 1: sync the configuration ---
    print("=" * 60)
    print("[chain] Step 1: sync config_base -> engine")
    print("=" * 60)
    sync_config.sync()

    # --- Step 1.5: create this run's session folder + the current symlink + open the STATUS trail ---
    ws_session = f"session_{time.strftime('%Y%m%d-%H%M%S')}"
    session_dir = os.path.join(WORKSPACE, ws_session)
    point_current(session_dir)                       # workspace/current -> this run's session folder
    if status_log:
        status_log.cycle_header(ws_session, f"dispatch/{dispatch_name}", 1)
        status_log.turn_header(dispatch_name, f"running dispatch/{dispatch_name} agent={agent_name}")

    # --- Step 2: execute (load + single shot; the instrumentation is inside run_role) ---
    print("=" * 60)
    print(f"[chain] Step 2: run dispatch/{dispatch_name} agent={agent_name}")
    print("=" * 60)

    result = run_role(dispatch_name, agent_name, user_message)

    # --- Step 3: write into workspace ---
    # ★The spoken prefix is recorded in pool/threads/{peer}.md instead.
    #   The output_*.md here is a prototype-era artefact for a human to read, not a memory
    #   carrier.
    text_response = result.get("text_response", "") or ""
    if text_response:
        workspace_dir = os.path.join(SHELL_ROOT, "workspace", dispatch_name)
        os.makedirs(workspace_dir, exist_ok=True)
        output_file = os.path.join(workspace_dir, f"output_{time.strftime('%Y%m%d_%H%M%S')}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text_response)
        print(f"[chain] output written to: {output_file}")

    if status_log:
        status_log.footer(f"archived under workspace/{ws_session}/. Took {result.get('duration_ms', 0)}ms")

    print("=" * 60)
    print(f"[chain] done. Took {result.get('duration_ms', 0)}ms · trail at workspace/current/STATUS.md")
    print("=" * 60)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="jev_voice_agent orchestration entry point")
    parser.add_argument("--dispatch", default="speaker", help="dispatch name")
    parser.add_argument("--agent", default="core", help="agent name")
    parser.add_argument("--message", default="", help="the user input message")
    parser.add_argument("--session", default=None,
                        help="session ID (★no effect: the engine does not support it; passing it warns)")

    args = parser.parse_args()

    run(
        dispatch_name=args.dispatch,
        user_message=args.message,
        agent_name=args.agent,
        session_id=args.session,
    )
