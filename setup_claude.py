"""One-time setup: configure MCP connection for ReCall.

Supports:
  python setup_claude.py claude    — Claude Desktop
  python setup_claude.py copilot   — VS Code Copilot
  python setup_claude.py           — both
"""
import json
import os
import sys

home = os.path.expanduser("~").replace("\\", "/")

mcp_env = {
    "NEUROPACK_DB_PATH": home + "/.neuropack/desktop_memories.db",
    "NEUROPACK_EMBEDDER_TYPE": "tfidf",
}


def setup_claude_desktop():
    """Configure Claude Desktop MCP."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        config_path = os.path.join(appdata, "Claude", "claude_desktop_config.json")
    else:
        config_path = os.path.expanduser(
            "~/Library/Application Support/Claude/claude_desktop_config.json"
        )

    mcp_entry = {
        "command": "python",
        "args": ["-m", "neuropack.mcp_server.server"],
        "env": mcp_env,
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw = f.read().strip()
            data = json.loads(raw) if raw else {}
    else:
        data = {}

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    data["mcpServers"]["recall"] = mcp_entry

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Claude Desktop] Config written to: {config_path}")
    print("  Restart Claude Desktop for changes to take effect.")


def setup_copilot():
    """Configure VS Code Copilot MCP."""
    # Write to .vscode/mcp.json in the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    vscode_dir = os.path.join(project_dir, ".vscode")
    config_path = os.path.join(vscode_dir, "mcp.json")

    mcp_config = {
        "servers": {
            "recall": {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "neuropack.mcp_server.server"],
                "env": mcp_env,
            }
        }
    }

    os.makedirs(vscode_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(mcp_config, f, indent=2)

    print(f"[VS Code Copilot] Config written to: {config_path}")
    print("  Make sure 'chat.mcp.enabled' is true in VS Code settings.")
    print("  Then use Copilot in Agent mode to access ReCall tools.")


if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if target in ("claude", "all"):
        setup_claude_desktop()
        print()

    if target in ("copilot", "vscode", "all"):
        setup_copilot()
        print()

    print("Then ask: 'Do you have access to memory tools?'")
