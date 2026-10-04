#!/usr/bin/env python3
"""
Generate MANIFEST.json — machine-readable system map.
Scans dispatch/ directory and writes MANIFEST.json.

Run manually:  python3 generate_manifest.py
"""

import os
import json
import glob
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DISPATCH_DIR = os.path.join(PROJECT_ROOT, "dispatch")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "MANIFEST.json")


def _scan_dispatches():
    """Scan dispatch/ directory. Returns dict of strategy_name → info."""
    dispatches = {}
    if not os.path.isdir(DISPATCH_DIR):
        return dispatches

    for name in sorted(os.listdir(DISPATCH_DIR)):
        strategy_dir = os.path.join(DISPATCH_DIR, name)
        dispatch_py = os.path.join(strategy_dir, "dispatch.py")
        if not os.path.isdir(strategy_dir) or not os.path.isfile(dispatch_py):
            continue

        config_path = os.path.join(strategy_dir, "dispatch_config.json")
        cfg = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        agents = list(cfg.get("agents", {}).keys())

        # Discover tools within dispatch strategy
        tools_dir = os.path.join(strategy_dir, "tools")
        tools = [os.path.basename(p) for p in
                 sorted(glob.glob(os.path.join(tools_dir, "*_tools.py")))] if os.path.isdir(tools_dir) else []

        dispatches[name] = {
            "config": f"dispatch/{name}/dispatch_config.json",
            "agents": agents,
            "tools": tools,
        }
    return dispatches


def generate():
    """Build and write MANIFEST.json."""
    manifest = {
        "framework": "IAF",
        "version": "2.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "structure": {
            "dispatch_dir": "dispatch/",
            "global_config": "config.json",
        },

        "dispatches": _scan_dispatches(),

        "conventions": {
            "tool_file_pattern": "*_tools.py",
            "tool_export_variable": "TOOLS",
            "context_strategy_dir": "context/",
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    generate()
    print(f"MANIFEST.json generated at {MANIFEST_PATH}")
