"""One-time setup: configure Claude Desktop MCP connection for ReCall."""
import json
import os
import sys

if sys.platform == "win32":
    appdata = os.environ.get("APPDATA", "")
    config_path = os.path.join(appdata, "Claude", "claude_desktop_config.json")
else:
    config_path = os.path.expanduser(
        "~/Library/Application Support/Claude/claude_desktop_config.json"
    )

home = os.path.expanduser("~").replace("\\", "/")

mcp_entry = {
    "command": "python",
    "args": ["-m", "neuropack.mcp_server.server"],
    "env": {
        "NEUROPACK_DB_PATH": home + "/.neuropack/desktop_memories.db",
        "NEUROPACK_EMBEDDER_TYPE": "tfidf",
    },
}

# Read existing or create new
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

print(f"Config written to: {config_path}")
print()
print("Restart Claude Desktop for changes to take effect.")
print()
print("Then ask Claude: 'Do you have access to memory tools?'")
