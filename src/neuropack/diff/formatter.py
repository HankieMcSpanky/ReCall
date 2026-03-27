from __future__ import annotations

from neuropack.diff.models import MemoryDiff, TimelineEntry


def format_diff_text(diff: MemoryDiff) -> str:
    """Format a MemoryDiff as CLI-friendly colored text output.

    Uses ANSI escape codes for color in terminal output.
    """
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    lines: list[str] = []

    # Header
    since_str = diff.since.strftime("%Y-%m-%d %H:%M")
    until_str = diff.until.strftime("%Y-%m-%d %H:%M")
    lines.append(f"{BOLD}Memory Diff: {since_str} -> {until_str}{RESET}")
    lines.append(
        f"  {GREEN}+{diff.stats.added} added{RESET}  "
        f"{YELLOW}~{diff.stats.modified} modified{RESET}  "
        f"{RED}-{diff.stats.deleted} deleted{RESET}"
    )
    lines.append("")

    # New memories
    if diff.new_memories:
        lines.append(f"{GREEN}{BOLD}+ New Memories ({len(diff.new_memories)}){RESET}")
        for m in diff.new_memories:
            tags_str = ", ".join(m.tags) if m.tags else "none"
            lines.append(f"  {GREEN}+{RESET} {m.id[:8]}  {m.l3_abstract}")
            lines.append(f"    {DIM}tags: {tags_str}  source: {m.source}{RESET}")
        lines.append("")

    # Updated memories
    if diff.updated_memories:
        lines.append(f"{YELLOW}{BOLD}~ Updated Memories ({len(diff.updated_memories)}){RESET}")
        for u in diff.updated_memories:
            lines.append(f"  {YELLOW}~{RESET} {u.id[:8]}")
            if u.old_l3 != u.new_l3:
                lines.append(f"    {RED}- {u.old_l3}{RESET}")
                lines.append(f"    {GREEN}+ {u.new_l3}{RESET}")
            added_tags = set(u.new_tags) - set(u.old_tags)
            removed_tags = set(u.old_tags) - set(u.new_tags)
            if added_tags:
                lines.append(f"    tags: {GREEN}+{', '.join(sorted(added_tags))}{RESET}")
            if removed_tags:
                lines.append(f"    tags: {RED}-{', '.join(sorted(removed_tags))}{RESET}")
        lines.append("")

    # Deleted memories
    if diff.deleted_ids:
        lines.append(f"{RED}{BOLD}- Deleted Memories ({len(diff.deleted_ids)}){RESET}")
        for did in diff.deleted_ids:
            lines.append(f"  {RED}-{RESET} {did[:8]}...")
        lines.append("")

    # Topics summary
    if diff.stats.topics_added:
        lines.append(f"{CYAN}New topics:{RESET} {', '.join(diff.stats.topics_added)}")
    if diff.stats.topics_removed:
        lines.append(f"{DIM}Removed topics:{RESET} {', '.join(diff.stats.topics_removed)}")

    return "\n".join(lines)


def format_diff_json(diff: MemoryDiff) -> dict:
    """Format a MemoryDiff as a structured JSON-serializable dict."""
    return {
        "since": diff.since.isoformat(),
        "until": diff.until.isoformat(),
        "stats": {
            "added": diff.stats.added,
            "modified": diff.stats.modified,
            "deleted": diff.stats.deleted,
            "topics_added": diff.stats.topics_added,
            "topics_removed": diff.stats.topics_removed,
        },
        "new_memories": [
            {
                "id": m.id,
                "l3_abstract": m.l3_abstract,
                "tags": m.tags,
                "created_at": m.created_at.isoformat(),
                "source": m.source,
            }
            for m in diff.new_memories
        ],
        "updated_memories": [
            {
                "id": u.id,
                "old_l3": u.old_l3,
                "new_l3": u.new_l3,
                "old_tags": u.old_tags,
                "new_tags": u.new_tags,
                "changed_at": u.changed_at.isoformat(),
            }
            for u in diff.updated_memories
        ],
        "deleted_ids": diff.deleted_ids,
    }


def format_timeline_text(entries: list[TimelineEntry]) -> str:
    """Format timeline entries as an ASCII timeline for CLI display."""
    if not entries:
        return "No activity in the selected range."

    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    lines: list[str] = []
    lines.append(f"{BOLD}Knowledge Timeline{RESET}")
    lines.append("")

    # Find max activity for bar scaling
    max_activity = max(
        (e.added + e.modified + e.deleted for e in entries),
        default=1,
    )
    bar_width = 40

    for entry in entries:
        total = entry.added + entry.modified + entry.deleted
        bar_len = max(1, int((total / max(max_activity, 1)) * bar_width))

        # Build colored bar segments
        add_len = max(0, int((entry.added / max(total, 1)) * bar_len))
        mod_len = max(0, int((entry.modified / max(total, 1)) * bar_len))
        del_len = max(0, bar_len - add_len - mod_len)

        bar = (
            f"{GREEN}{'=' * add_len}{RESET}"
            f"{YELLOW}{'=' * mod_len}{RESET}"
            f"{RED}{'=' * del_len}{RESET}"
        )

        label = f"{entry.period_label:>15}"
        counts = f"+{entry.added} ~{entry.modified} -{entry.deleted}"
        tags = f"  {DIM}[{', '.join(entry.top_tags[:3])}]{RESET}" if entry.top_tags else ""

        lines.append(f"  {label}  |{bar}| {counts}{tags}")

    lines.append("")
    lines.append(f"  {GREEN}={RESET} added  {YELLOW}={RESET} modified  {RED}={RESET} deleted")

    return "\n".join(lines)
