"""Shell integration for NeuroPack — captures command+output pairs into memory."""

from __future__ import annotations

import os
import re
from pathlib import Path

from neuropack.core.store import MemoryStore

# Commands that are too trivial to store
TRIVIAL_COMMANDS = frozenset({
    "cd", "ls", "ll", "la", "clear", "exit", "quit", "pwd", "echo",
    "cat", "cls", "dir", "history", "whoami", "date", "true", "false",
    ":", "fg", "bg", "jobs", "popd", "pushd", "dirs", "type", "which",
    "alias", "unalias", "set", "unset", "export", "source", ".",
})

# Minimum command length to be considered interesting
MIN_COMMAND_LENGTH = 5


def _is_interesting(command: str) -> bool:
    """Return True if the command is worth storing in memory."""
    stripped = command.strip()
    if len(stripped) < MIN_COMMAND_LENGTH:
        return False

    # Extract the base command (first word)
    base = stripped.split()[0] if stripped else ""
    # Strip path prefixes (e.g., /usr/bin/ls -> ls)
    base = os.path.basename(base)

    if base in TRIVIAL_COMMANDS:
        # Still store if the command has pipes, redirections, or many flags
        if "|" in stripped or ">" in stripped or "<" in stripped:
            return True
        # Count flags
        parts = stripped.split()
        flags = [p for p in parts[1:] if p.startswith("-")]
        if len(flags) >= 2:
            return True
        return False

    return True


def _directory_tag(cwd: str) -> str:
    """Create a short tag from a directory path."""
    if not cwd:
        return ""
    name = Path(cwd).name
    # Sanitize to tag-safe characters
    name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    return f"dir:{name}" if name else ""


def log_command(
    command: str,
    exit_code: int = 0,
    cwd: str = "",
    store: MemoryStore | None = None,
) -> None:
    """Store a shell command as a NeuroPack memory.

    Designed to be fast — opens store, writes, closes quickly.
    Silently ignores trivial or uninteresting commands.
    """
    command = command.strip()
    if not _is_interesting(command):
        return

    if store is None:
        from neuropack.config import NeuropackConfig

        config = NeuropackConfig()
        store = MemoryStore(config)
        store.initialize()
        should_close = True
    else:
        should_close = False

    try:
        effective_cwd = cwd or os.getcwd()

        # Build memory content
        content = f"Shell: {command}\nDirectory: {effective_cwd}\nExit: {exit_code}"

        # Build tags
        tags = ["shell-command"]
        tags.append(f"exit-{exit_code}")
        dir_tag = _directory_tag(effective_cwd)
        if dir_tag:
            tags.append(dir_tag)

        # Source based on directory basename
        dir_name = Path(effective_cwd).name if effective_cwd else "shell"
        source = f"shell:{dir_name}"

        store.store(
            content=content,
            tags=tags,
            source=source,
            priority=0.3 if exit_code == 0 else 0.5,
        )
    except Exception:
        # Never interfere with the user's shell
        pass
    finally:
        if should_close:
            store.close()


def search_commands(
    query: str,
    store: MemoryStore | None = None,
    limit: int = 10,
) -> list:
    """Search past shell commands in NeuroPack memory.

    Convenience wrapper around recall filtered to the shell-command tag.
    """
    if store is None:
        from neuropack.config import NeuropackConfig

        config = NeuropackConfig()
        store = MemoryStore(config)
        store.initialize()
        should_close = True
    else:
        should_close = False

    try:
        results = store.recall(
            query=query,
            limit=limit,
            tags=["shell-command"],
        )
        return results
    except Exception:
        return []
    finally:
        if should_close:
            store.close()


def generate_bash_hook() -> str:
    """Return a bash script snippet for shell integration.

    Users source this in .bashrc via: eval "$(np shell-init --shell bash)"
    """
    return r'''# NeuroPack shell integration (bash)
_neuropack_preexec() {
    _NP_LAST_CMD="$1"
}
_neuropack_precmd() {
    local exit_code=$?
    if [ -n "$_NP_LAST_CMD" ]; then
        np shell-log "$_NP_LAST_CMD" "$exit_code" &>/dev/null &
        disown 2>/dev/null
        _NP_LAST_CMD=""
    fi
}
trap '_neuropack_preexec "$BASH_COMMAND"' DEBUG
PROMPT_COMMAND="_neuropack_precmd;${PROMPT_COMMAND}"
'''


def generate_zsh_hook() -> str:
    """Return a zsh script snippet for shell integration.

    Users source this in .zshrc via: eval "$(np shell-init --shell zsh)"
    """
    return r'''# NeuroPack shell integration (zsh)
_neuropack_preexec() { _NP_LAST_CMD="$1" }
_neuropack_precmd() {
    local exit_code=$?
    if [[ -n "$_NP_LAST_CMD" ]]; then
        np shell-log "$_NP_LAST_CMD" "$exit_code" &>/dev/null &
        disown 2>/dev/null
        _NP_LAST_CMD=""
    fi
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec _neuropack_preexec
add-zsh-hook precmd _neuropack_precmd
'''


def generate_powershell_hook() -> str:
    """Return a PowerShell profile snippet for shell integration.

    Users add this to $PROFILE via: np shell-init --shell powershell >> $PROFILE
    """
    return r'''# NeuroPack shell integration (PowerShell)
$_NeuroPack_Original_Prompt = $function:prompt
function prompt {
    $lastExit = $LASTEXITCODE
    $lastCmd = (Get-History -Count 1).CommandLine
    if ($lastCmd) {
        Start-Job -ScriptBlock {
            param($cmd, $code)
            np shell-log $cmd $code
        } -ArgumentList $lastCmd, $lastExit | Out-Null
    }
    & $_NeuroPack_Original_Prompt
}
'''
