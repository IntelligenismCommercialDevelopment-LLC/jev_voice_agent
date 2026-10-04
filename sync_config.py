"""
sync_config.py -- push the configuration in config_base into IAF_AI.

Responsibilities:
1. config_base/config.json -> IAF_AI/config.json (a straight copy)
2. config_base/{dispatch}/dispatch_config.json -> IAF_AI/dispatch/{dispatch}/dispatch_config.json
   - translate the relative paths in context_files (shell-root-relative -> agent_dir-relative)
   - absolute paths and prefixed paths (wiki:, session:, briefs:) are left alone
"""

import json
import os
import shutil

SHELL_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_BASE = os.path.join(SHELL_ROOT, "config_base")
IAF_AI_ROOT = os.path.join(SHELL_ROOT, "IAF_AI")
IAF_AI_DISPATCH = os.path.join(IAF_AI_ROOT, "dispatch")
SHARED_TOOLS = os.path.join(SHELL_ROOT, "tools", "_shared")   # the canonical copy of the fenced tools shared by per-role dispatches

# How deep agent_dir sits below shell_root: IAF_AI/dispatch/{name}/ = 3 levels
AGENT_DIR_TO_SHELL_ROOT = os.path.join("..", "..", "..")


def _translate_path(path: str) -> str:
    """
    Translate a shell-root-relative path into an agent_dir-relative one.

    Rules:
    - absolute paths (starting with /) -> not translated
    - prefixed paths (wiki:, session:, briefs:) -> not translated
    - relative paths -> prefixed with ../../../ so that they resolve to the shell root when
      followed from agent_dir
    """
    if os.path.isabs(path):
        return path
    if ":" in path and not path.startswith("./"):
        return path
    return os.path.join(AGENT_DIR_TO_SHELL_ROOT, path)


def sync():
    """Sync config_base -> IAF_AI."""
    if not os.path.isdir(CONFIG_BASE):
        print("[sync_config] config_base/ does not exist; skipping.")
        return

    # --- 1. the global config.json ---
    global_config_src = os.path.join(CONFIG_BASE, "config.json")
    global_config_dst = os.path.join(IAF_AI_ROOT, "config.json")

    if os.path.isfile(global_config_src):
        shutil.copy2(global_config_src, global_config_dst)
        print(f"[sync_config] config.json → {global_config_dst}")

    # --- 2. each dispatch's dispatch_config.json ---
    for dispatch_name in os.listdir(CONFIG_BASE):
        if dispatch_name.startswith("_"):        # archive directories such as _retired/ are not synced
            continue
        config_dir = os.path.join(CONFIG_BASE, dispatch_name)
        if not os.path.isdir(config_dir):
            continue

        engine_dispatch_dir = os.path.join(IAF_AI_DISPATCH, dispatch_name)
        if not os.path.isdir(engine_dispatch_dir):
            print(f"[sync_config] warning: engine dispatch '{dispatch_name}' does not exist; skipping.")
            continue

        src_path = os.path.join(config_dir, "dispatch_config.json")
        dst_path = os.path.join(engine_dispatch_dir, "dispatch_config.json")

        if not os.path.isfile(src_path):
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # translate the context_files paths
        for agent_name, agent_conf in config.get("agents", {}).items():
            if "context_files" in agent_conf:
                agent_conf["context_files"] = [
                    _translate_path(p) for p in agent_conf["context_files"]
                ]

        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[sync_config] {dispatch_name}/dispatch_config.json → {dst_path}")

        # If this dispatch uses free_search (the per-role fenced tools), lay the shared
        # explore_tools.py + status_log.py into its tools/
        uses_free_search = any(
            a.get("free_search") for a in config.get("agents", {}).values()
        )
        if uses_free_search:
            _sync_tools(engine_dispatch_dir)

    print("[sync_config] done.")


def _sync_tools(engine_dispatch_dir: str):
    """Copy tools/_shared/explore_tools.py into a dispatch's engine-side tools/ directory and
    clear out leftover marker probes and stale tools.
    The per-role fenced tools are one single piece of code; behaviour is driven by each
    dispatch_config.json's free_search block.
    status_log.py is laid down beside it so explore_tools can import it as a sibling (the
    live per-event STATUS trail; it degrades silently when missing)."""
    src = os.path.join(SHARED_TOOLS, "explore_tools.py")
    if not os.path.isfile(src):
        print(f"[sync_config] warning: shared tool {src} not found; skipping the tools sync.")
        return
    tools_dir = os.path.join(engine_dispatch_dir, "tools")
    os.makedirs(tools_dir, exist_ok=True)
    # Remove marker probes (marker_*_tools.py) left over from isolation testing, to avoid contamination
    for stale in os.listdir(tools_dir):
        if stale.startswith("marker_") and stale.endswith("_tools.py"):
            os.remove(os.path.join(tools_dir, stale))
    shutil.copy2(src, os.path.join(tools_dir, "explore_tools.py"))
    print(f"[sync_config]   tools/explore_tools.py → {tools_dir}")
    slog = os.path.join(SHARED_TOOLS, "status_log.py")
    if os.path.isfile(slog):
        shutil.copy2(slog, os.path.join(tools_dir, "status_log.py"))
        print(f"[sync_config]   tools/status_log.py → {tools_dir}")


if __name__ == "__main__":
    sync()
