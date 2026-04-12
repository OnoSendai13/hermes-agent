"""
Cowork tools for Hermes Agent.

Exposes Cowork plugin capabilities as Hermes tools:
- Process monitoring (psutil)
- Web agent (MolmoWeb via subprocess)
- Context file read/write for Claude Code <-> Hermes sharing

The Cowork daemon doesn't need to be running for these tools -- they call
the underlying libraries directly.
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

# ==============================================================================
# Paths
# ==============================================================================

COWORK_DIR = Path.home() / ".cowork"
WORKSPACE_DIR = COWORK_DIR / "workspace"
CONTEXT_FILE = WORKSPACE_DIR / "context.json"
PLUGIN_DIR = COWORK_DIR / "plugins"

# Ensure workspace exists
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Shared context helpers
# ==============================================================================

def _read_context() -> dict:
    """Read the shared context file."""
    if not CONTEXT_FILE.exists():
        return {"hermes_context": {}, "claude_code_context": {}}
    try:
        with open(CONTEXT_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"hermes_context": {}, "claude_code_context": {}}


def _write_context(data: dict) -> None:
    """Write the shared context file atomically."""
    tmp = CONTEXT_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(CONTEXT_FILE)


# ==============================================================================
# Tool schemas
# ==============================================================================

STATUS_SCHEMA = {
    "name": "cowork_status",
    "description": "Check Cowork system status -- daemon, Redis, plugins, Claude Code, and shared context summary.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

PROCESS_LIST_SCHEMA = {
    "name": "cowork_process_list",
    "description": "List top processes by CPU usage. Returns PID, name, CPU%, RAM%, and command line.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max processes to return (default: 15).",
                "default": 15,
            },
        },
        "required": [],
    },
}

SYSTEM_RESOURCES_SCHEMA = {
    "name": "cowork_system_resources",
    "description": "Get system resource summary: CPU usage, RAM (used/total), disk, and boot time.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

CONTEXT_READ_SCHEMA = {
    "name": "cowork_context_read",
    "description": "Read the shared Claude Code <-> Hermes context file. Shows what each agent has written.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

CONTEXT_WRITE_SCHEMA = {
    "name": "cowork_context_write",
    "description": "Write to the shared Claude Code <-> Hermes context. Used to pass user goals to Claude Code, or store results for Claude Code to read.",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Which agent is writing: 'hermes' or 'claude_code'.",
                "enum": ["hermes", "claude_code"],
            },
            "key": {
                "type": "string",
                "description": "Context key (e.g., 'user_goal', 'last_task', 'workspace_state').",
            },
            "value": {
                "description": "JSON-serializable value to store.",
            },
        },
        "required": ["agent", "key", "value"],
    },
}

WEB_AGENT_TASK_SCHEMA = {
    "name": "cowork_web_task",
    "description": "Run an autonomous web task using MolmoWeb (local Chromium, no API key needed). Give a natural-language goal like 'find the cheapest flight Paris to Rome for Friday' and the agent navigates, clicks, types, and reads pages to complete it. Returns step-by-step trajectory and final answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural-language description of the goal.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Max browser steps (default: 15, complex tasks may need 30+).",
                "default": 15,
            },
        },
        "required": ["task"],
    },
}

WEB_AGENT_SNAPSHOT_SCHEMA = {
    "name": "cowork_web_snapshot",
    "description": "Get an accessibility tree snapshot of the current browser page (if MolmoWeb is running). Use after cowork_web_task to see the page state.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

RUN_CODE_TASK_SCHEMA = {
    "name": "cowork_run_code_task",
    "description": "Trigger Claude Code to perform a coding task. The task is queued and executed asynchronously — results are written to the shared context.json file. Poll with cowork_context_read to get the result when goal_status is 'done'. This is the primary tool for delegating coding work to Claude Code.",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural-language description of the coding task for Claude Code.",
            },
        },
        "required": ["task"],
    },
}

SCREENSHOT_CAPTURE_SCHEMA = {
    "name": "cowork_screenshot_capture",
    "description": "Capture a screenshot now. Returns the image path and size. Screenshots are stored locally only in ~/.cowork/screenshots/.",
    "parameters": {
        "type": "object",
        "properties": {
            "output_path": {
                "type": "string",
                "description": "Optional output path for the screenshot. Defaults to ~/.cowork/screenshots/screenshot_TIMESTAMP.png.",
            },
        },
        "required": [],
    },
}

SCREENSHOT_LIST_SCHEMA = {
    "name": "cowork_screenshot_list",
    "description": "List recent screenshots stored in ~/.cowork/screenshots/.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max screenshots to return (default: 10).",
                "default": 10,
            },
        },
        "required": [],
    },
}

# ==============================================================================
# Tool implementations
# ==============================================================================

def _handle_cowork_status(args, **kw) -> str:
    """Check cowork system status."""
    daemon_running = False
    redis_ok = False
    claude_ok = False

    # Check daemon (rough: check if coworkd process is running)
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmdline = " ".join(p.info["cmdline"] or [])
            if "coworkd" in cmdline:
                daemon_running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Check Redis
    try:
        import redis.asyncio as redis
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        r = loop.run_until_complete(
            redis.from_url("redis://10.255.255.254:6379", decode_responses=True)
        )
        loop.run_until_complete(r.ping())
        redis_ok = True
        loop.run_until_complete(r.aclose())
        loop.close()
    except Exception:
        pass

    # Check Claude Code
    try:
        result = subprocess.run(
            ["claude-code", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            claude_ok = True
            claude_version = result.stdout.strip()
        else:
            claude_version = "found but error"
    except Exception:
        claude_version = "not found"

    # Context summary
    ctx = _read_context()

    return json.dumps({
        "daemon_running": daemon_running,
        "redis": "connected" if redis_ok else "not reachable",
        "claude_code": claude_ok,
        "claude_version": claude_version if claude_ok else claude_version,
        "context_file": str(CONTEXT_FILE),
        "context": ctx,
        "cowork_version": ctx.get("cowork", {}).get("version", "unknown"),
        "last_update": ctx.get("cowork", {}).get("last_update"),
    }, indent=2)


def _handle_process_list(args, **kw) -> str:
    """List top processes by CPU."""
    limit = args.get("limit", 15)
    processes = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "cmdline"]):
        try:
            info = p.info
            info["cpu_percent"] = info["cpu_percent"] or 0.0
            info["memory_percent"] = info["memory_percent"] or 0.0
            processes.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    processes.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
    top = processes[:limit]

    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "count": len(top),
        "processes": top,
    }, indent=2)


def _handle_system_resources(args, **kw) -> str:
    """Get system resource summary."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    boot_time = datetime.fromtimestamp(psutil.boot_time()).isoformat()

    # Network interfaces
    net = {}
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == 2:  # IPv4
                    net[iface] = addr.address
    except Exception:
        pass

    return json.dumps({
        "cpu": {
            "percent": cpu_percent,
            "count": cpu_count,
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
            "percent": disk.percent,
        },
        "boot_time": boot_time,
        "network": net,
    }, indent=2)


def _handle_context_read(args, **kw) -> str:
    """Read shared context file."""
    ctx = _read_context()
    return json.dumps(ctx, indent=2)


def _handle_context_write(args, **kw) -> str:
    """Write to shared context file."""
    agent = args.get("agent")
    key = args.get("key")
    value = args.get("value")
    if agent not in ("hermes", "claude_code"):
        return json.dumps({"error": f"Invalid agent: {agent}"})
    if not key:
        return json.dumps({"error": "key is required"})
    if value is None:
        return json.dumps({"error": "value is required"})

    ctx = _read_context()
    section = f"{agent}_context"
    if section not in ctx:
        ctx[section] = {}
    ctx[section][key] = value
    ctx["cowork"] = {"version": "1.0.0", "last_update": datetime.now().isoformat()}

    _write_context(ctx)

    return json.dumps({
        "ok": True,
        "agent": agent,
        "key": key,
        "value": value,
        "context_file": str(CONTEXT_FILE),
    })


def _handle_web_task(args, **kw) -> str:
    """Run MolmoWeb agent task via subprocess."""
    task = args.get("task")
    max_steps = args.get("max_steps", 15)
    if not task:
        return json.dumps({"error": "task is required"})

    # Check if molmoweb is available
    molmoweb_lib = COWORK_DIR / "lib" / "molmoweb"
    if not molmoweb_lib.exists():
        return json.dumps({
            "error": "MolmoWeb not installed. Run: git clone --depth=1 https://github.com/allenai/molmoweb ~/.cowork/lib/molmoweb"
        })

    # Write task to context for Claude Code visibility
    ctx = _read_context()
    ctx["hermes_context"]["user_goal"] = task
    ctx["hermes_context"]["goal_timestamp"] = datetime.now().isoformat()
    ctx["cowork"] = {"version": "1.0.0", "last_update": datetime.now().isoformat()}
    _write_context(ctx)

    # Run molmo_agent via Python - invoke the plugin directly
    # We use a script that imports and runs the molmo_agent plugin
    script = f"""
import sys
import json
import asyncio
sys.path.insert(0, '{str(PLUGIN_DIR)}')
sys.path.insert(0, '{str(molmoweb_lib)}')

from base import CoworkContext
import molmo_agent as _ma
Plugin = _ma.Plugin

async def run():
    ctx = CoworkContext(
        workspace={repr(str(WORKSPACE_DIR))},
        memory={{}},
        processes=[],
        files_changed=[],
        user_info={{}},
    )
    plugin = Plugin(ctx)
    await plugin.on_start()
    if not plugin._ready:
        return {{"error": "MolmoWeb not ready"}}

    # Run the web task
    result = await plugin.run_task({repr(task)}, max_steps={max_steps})

    # Also capture a screenshot of the final page state
    screenshot_result = await plugin.take_screenshot()
    try:
        screenshot_data = json.loads(screenshot_result)
    except Exception:
        screenshot_data = {{"error": screenshot_result}}

    # Parse the task result and add screenshot info
    try:
        result_data = json.loads(result)
        result_data["screenshot"] = screenshot_data
        return result_data
    except Exception:
        return {{"task_result": result, "screenshot": screenshot_data}}

print(json.dumps(asyncio.run(run())))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
            cwd=str(WORKSPACE_DIR),
        )
        if result.returncode != 0:
            return json.dumps({
                "error": f"MolmoWeb failed: {result.stderr[:500]}",
                "stdout": result.stdout[:500],
            })
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Task timed out after 5 minutes"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_run_code_task(args, **kw) -> str:
    """
    Queue a Claude Code task via context.json.
    The daemon's task_orchestrator picks it up and executes it.
    """
    task = args.get("task")
    if not task:
        return json.dumps({"error": "task is required"})

    ctx = _read_context()
    ctx["hermes_context"]["user_goal"] = task
    ctx["hermes_context"]["goal_status"] = "pending"
    ctx["hermes_context"]["goal_source"] = "hermes"
    ctx["hermes_context"]["goal_created_at"] = datetime.now().isoformat()
    ctx["hermes_context"].pop("goal_result", None)
    ctx["hermes_context"].pop("goal_error", None)
    ctx["hermes_context"].pop("goal_completed_at", None)
    ctx["claude_code_context"]["last_task"] = ""
    ctx["claude_code_context"].pop("task_result", None)
    ctx["claude_code_context"].pop("task_completed_at", None)
    ctx["cowork"] = {"version": "1.0.0", "last_update": datetime.now().isoformat()}
    _write_context(ctx)

    return json.dumps({
        "ok": True,
        "task": task,
        "message": "Task queued. Poll cowork_context_read until goal_status is 'done'.",
        "context_file": str(CONTEXT_FILE),
    })


def _handle_web_snapshot(args, **kw) -> str:
    """Get current page snapshot from MolmoWeb."""
    ctx = _read_context()
    hermes_ctx = ctx.get("hermes_context", {})
    last_goal = hermes_ctx.get("user_goal", "")
    last_update = hermes_ctx.get("goal_timestamp", "")
    return json.dumps({
        "note": "Snapshot requires Cowork daemon running. Start with: python ~/.cowork/coworkd.py",
        "last_user_goal": last_goal,
        "last_update": last_update,
    })


def _handle_screenshot_capture(args, **kw) -> str:
    """Capture a screenshot using mss with Xvfb fallback."""
    output_path = args.get("output_path")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = Path.home() / ".cowork" / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    if output_path:
        path = Path(output_path).expanduser()
    else:
        path = screenshot_dir / f"screenshot_{timestamp}.png"

    try:
        import mss
        import subprocess
        import os

        # Check if we have a display
        has_display = bool(os.environ.get("DISPLAY"))

        if not has_display:
            # Try Xvfb fallback
            xvfb = subprocess.run(
                ["which", "xvfb-run"], capture_output=True, text=True
            )
            if xvfb.returncode == 0:
                # Wrap mss call in xvfb-run
                script = f"""
import mss
sct = mss.mss()
sct.shot(output={repr(str(path))})
"""
                result = subprocess.run(
                    ["xvfb-run", "-a",
                     sys.executable, "-c", script],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return json.dumps({
                        "status": "captured",
                        "path": str(path),
                        "size": path.stat().st_size if path.exists() else 0,
                        "backend": "xvfb+mss",
                    })
                else:
                    return json.dumps({
                        "error": f"xvfb-run failed: {result.stderr[:200]}"
                    })
            else:
                return json.dumps({
                    "error": "No DISPLAY set and xvfb-run not available. "
                             "Install xvfb: sudo apt install xvfb"
                })

        # Native mss with real display
        with mss.mss() as sct:
            sct.shot(output=str(path))

        return json.dumps({
            "status": "captured",
            "path": str(path),
            "size": path.stat().st_size,
            "backend": "mss",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_screenshot_list(args, **kw) -> str:
    """List recent screenshots."""
    limit = args.get("limit", 10)
    screenshot_dir = Path.home() / ".cowork" / "screenshots"
    if not screenshot_dir.exists():
        return json.dumps({"screenshots": []})

    screenshots = sorted(
        screenshot_dir.glob("screenshot_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    return json.dumps({
        "screenshots": [
            {
                "name": s.name,
                "path": str(s),
                "size": s.stat().st_size,
                "modified": datetime.fromtimestamp(s.stat().st_mtime).isoformat(),
            }
            for s in screenshots
        ]
    })


# ==============================================================================
# Registry
# ==============================================================================

from tools.registry import registry

registry.register(
    name="cowork_status",
    toolset="cowork",
    schema=STATUS_SCHEMA,
    handler=_handle_cowork_status,
    emoji="☤",
)

registry.register(
    name="cowork_process_list",
    toolset="cowork",
    schema=PROCESS_LIST_SCHEMA,
    handler=_handle_process_list,
    emoji="⚡",
)

registry.register(
    name="cowork_system_resources",
    toolset="cowork",
    schema=SYSTEM_RESOURCES_SCHEMA,
    handler=_handle_system_resources,
    emoji="🖥️",
)

registry.register(
    name="cowork_context_read",
    toolset="cowork",
    schema=CONTEXT_READ_SCHEMA,
    handler=_handle_context_read,
    emoji="📄",
)

registry.register(
    name="cowork_context_write",
    toolset="cowork",
    schema=CONTEXT_WRITE_SCHEMA,
    handler=_handle_context_write,
    emoji="✍️",
)

registry.register(
    name="cowork_web_task",
    toolset="cowork",
    schema=WEB_AGENT_TASK_SCHEMA,
    handler=_handle_web_task,
    emoji="🌐",
)

registry.register(
    name="cowork_web_snapshot",
    toolset="cowork",
    schema=WEB_AGENT_SNAPSHOT_SCHEMA,
    handler=_handle_web_snapshot,
    emoji="📸",
)

registry.register(
    name="cowork_run_code_task",
    toolset="cowork",
    schema=RUN_CODE_TASK_SCHEMA,
    handler=_handle_run_code_task,
    emoji="🤖",
)

registry.register(
    name="cowork_screenshot_capture",
    toolset="cowork",
    schema=SCREENSHOT_CAPTURE_SCHEMA,
    handler=_handle_screenshot_capture,
    emoji="📷",
)

registry.register(
    name="cowork_screenshot_list",
    toolset="cowork",
    schema=SCREENSHOT_LIST_SCHEMA,
    handler=_handle_screenshot_list,
    emoji="🖼️",
)
