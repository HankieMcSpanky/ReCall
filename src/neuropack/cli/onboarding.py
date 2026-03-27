"""Onboarding: np init wizard and np doctor diagnostics."""
from __future__ import annotations

import os
import time
from pathlib import Path

import click

from neuropack.config import NeuropackConfig
from neuropack.core.store import MemoryStore


EXAMPLE_MEMORIES = [
    "NeuroPack uses 3-level compression: L3 (abstract), L2 (key facts), L1 (full zstd-compressed text).",
    "Namespaces enable multi-agent memory sharing -- each agent gets its own namespace but can read from shared.",
    "The knowledge graph extracts entities and relationships from memories automatically.",
]


def run_init(config: NeuropackConfig) -> None:
    """Interactive setup wizard."""
    click.echo("")
    click.echo("  NeuroPack - AI Memory Layer")
    click.echo("  Local, private memory for AI agents with 3-level compression.")
    click.echo("")

    db_path = Path(config.db_path).expanduser()

    # Step 1: Database
    click.echo("[1/4] Database")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(config)
    store.initialize()
    click.echo(f"  Created {db_path}  [OK]")
    click.echo("")

    # Step 1b: Embedder choice
    click.echo("  Embedder:")
    click.echo("    1. TF-IDF (default, no extra deps)")
    click.echo("    2. Sentence-Transformer (384d dense, requires torch)")
    emb_choice = click.prompt("  Choice", type=click.IntRange(1, 2), default=1)
    if emb_choice == 2:
        try:
            from neuropack.embeddings.sentence_transformer import SentenceTransformerEmbedder  # noqa: F401

            click.echo("  sentence-transformers detected  [OK]")
            _write_config_env(db_path.parent, embedder_type="sentence-transformer")
        except ImportError:
            click.echo("  sentence-transformers not installed. Using TF-IDF.")
            click.echo("  Install with: pip install 'neuropack[transformers]'")
    click.echo("")

    # Step 2: LLM Configuration
    click.echo("[2/4] LLM Configuration (optional - for smarter compression)")
    if click.confirm("  Configure an LLM provider?", default=False):
        _configure_llm(store)
    else:
        click.echo("  Skipped (using extractive compression)")
    click.echo("")

    # Step 2b: Encryption
    click.echo("  Encryption at rest (optional - encrypts stored content)")
    if click.confirm("  Enable encryption?", default=False):
        from neuropack.storage.encryption import FieldEncryptor

        key = FieldEncryptor.generate_key()
        _write_config_env(db_path.parent, encryption_key=key)
        click.echo(f"  Generated key: {key[:8]}...{key[-4:]}")
        click.echo("  IMPORTANT: Back up NEUROPACK_ENCRYPTION_KEY from config.env")
        click.echo("  If you lose this key, encrypted data is unrecoverable.")
    else:
        click.echo("  Skipped (data stored in plaintext)")
    click.echo("")

    # Step 3: API Security
    click.echo("[3/4] API Security (optional - for HTTP server)")
    if click.confirm("  Set auth token?", default=False):
        token = click.prompt("  Auth token", hide_input=True)
        if token:
            _write_config_env(db_path.parent, auth_token=token)
            click.echo("  Saved to config.env")
        else:
            click.echo("  Skipped (server will accept all requests)")
    else:
        click.echo("  Skipped (server will accept all requests)")
    click.echo("")

    # Step 4: Example Memories
    click.echo("[4/4] Example Memories")
    click.echo("  Storing 3 example memories to get you started...")
    for text in EXAMPLE_MEMORIES:
        store.store(content=text, tags=["example", "getting-started"], source="np init")
        short = text[:50] + "..." if len(text) > 50 else text
        click.echo(f'  - "{short}" [stored]')
    click.echo("")

    # Done
    click.echo("  Ready! Here's what to try:")
    click.echo("")
    click.echo('    np recall "compression"     # Search memories')
    click.echo("    np list                     # See all memories")
    click.echo('    np store "your text" -t tag # Save something')
    click.echo("    np llm list                 # See configured LLMs")
    click.echo("    np serve                    # Start web dashboard")
    click.echo("    np --help                   # All commands")
    click.echo("")

    store.close()


def _configure_llm(store: MemoryStore) -> None:
    """Interactive LLM provider setup."""
    click.echo("  Provider:")
    click.echo("    1. OpenAI (GPT-4o-mini)")
    click.echo("    2. Anthropic (Claude Haiku)")
    click.echo("    3. Google (Gemini Flash)")
    click.echo("    4. OpenAI-compatible (Ollama, vLLM, Together, Groq, Azure...)")

    choice = click.prompt("  Choice", type=click.IntRange(1, 4), default=1)
    provider_map = {1: "openai", 2: "anthropic", 3: "gemini", 4: "openai-compatible"}
    provider = provider_map[choice]

    base_url = ""
    if choice == 4:
        base_url = click.prompt("  Base URL", default="http://localhost:11434/v1")

    api_key = click.prompt("  API key (or press Enter to skip)", default="", show_default=False)

    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-2.0-flash",
        "openai-compatible": "llama3.2",
    }
    model = click.prompt("  Model name", default=default_models.get(provider, ""))

    name = click.prompt("  Config name", default=provider if choice != 4 else "local")

    from neuropack.llm.models import LLMConfig

    llm_config = LLMConfig(
        name=name,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        is_default=True,
    )

    # Test connection
    click.echo("  Testing connection...", nl=False)
    store._llm_registry.add(llm_config)
    result = store._llm_registry.test_connection(name)
    if result["ok"]:
        click.echo(f' OK (response in {result["time_s"]}s)')
    else:
        click.echo(f' WARN ({result["error"]})')
        click.echo("  Saved anyway -- you can test later with: np llm test " + name)

    click.echo(f'  Saved as "{name}" (default)')


def _write_config_env(
    neuropack_dir: Path,
    auth_token: str = "",
    embedder_type: str = "",
    encryption_key: str = "",
) -> None:
    """Write/append config.env to the neuropack directory."""
    env_path = neuropack_dir / "config.env"
    # Read existing lines
    existing: list[str] = []
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").strip().splitlines()

    def _set_or_add(key: str, value: str) -> None:
        for i, line in enumerate(existing):
            if line.startswith(f"{key}=") or line.startswith(f"# {key}="):
                existing[i] = f"{key}={value}" if value else f"# {key}="
                return
        if value:
            existing.append(f"{key}={value}")

    if not existing:
        existing.append("# NeuroPack Configuration (generated by np init)")

    if auth_token:
        _set_or_add("NEUROPACK_AUTH_TOKEN", auth_token)
    if embedder_type:
        _set_or_add("NEUROPACK_EMBEDDER_TYPE", embedder_type)
    if encryption_key:
        _set_or_add("NEUROPACK_ENCRYPTION_KEY", encryption_key)

    env_path.write_text("\n".join(existing) + "\n", encoding="utf-8")


def run_doctor(store: MemoryStore) -> None:
    """Run diagnostics and report health."""
    click.echo("")
    click.echo("  NeuroPack Health Check")
    click.echo("")

    issues: list[str] = []
    db_path = Path(store.config.db_path).expanduser()

    # Embedder type
    click.echo(f"  Embedder .............. {store.config.embedder_type} ({store._embedder.dim}d)")

    # Encryption status
    enc_status = "enabled" if store.config.encryption_key else "disabled"
    click.echo(f"  Encryption ............ {enc_status}")

    # Database
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        click.echo(f"  Database .............. OK  ({db_path}, {size_kb:.0f} KB)")
    else:
        click.echo(f"  Database .............. FAIL  ({db_path} not found)")
        issues.append("Database file not found. Run: np init")

    # Schema
    try:
        s = store.stats()
        click.echo("  Schema ................ OK  (v3 with namespaces + knowledge graph)")
    except Exception as e:
        click.echo(f"  Schema ................ FAIL  ({e})")
        issues.append("Schema error. Run: np init")
        s = None

    # Memory count
    if s is not None:
        ns_list = store.list_namespaces()
        ns_count = len(ns_list) if ns_list else 0
        click.echo(f"  Memory count .......... {s.total_memories} memories across {ns_count} namespaces")

    # Knowledge graph
    kg = store.knowledge_graph_stats()
    entities = kg.get("entities", 0)
    relationships = kg.get("relationships", 0)
    click.echo(f"  Knowledge graph ....... {entities} entities, {relationships} relationships")

    # API keys
    api_keys = store._api_key_manager.list_keys()
    click.echo(f"  API keys .............. {len(api_keys)} active")

    # LLM configs
    llm_configs = store._llm_registry.list_all()
    if not llm_configs:
        click.echo("  LLM ................... None configured")
    else:
        for llm_cfg in llm_configs:
            result = store._llm_registry.test_connection(llm_cfg.name)
            if result["ok"]:
                click.echo(f'  LLM ({llm_cfg.name}) .... OK  (response in {result["time_s"]}s)')
            else:
                click.echo(f'  LLM ({llm_cfg.name}) .... FAIL ({result["error"]})')
                issues.append(
                    f'LLM "{llm_cfg.name}": {result["error"]}. Run: np llm test {llm_cfg.name}'
                )

    # Disk usage
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        click.echo(f"  Disk usage ............ {size_mb:.1f} MB (database)")

    click.echo("")
    if issues:
        click.echo(f"  {len(issues)} issue(s) found:")
        for issue in issues:
            click.echo(f"    - {issue}")
    else:
        click.echo("  All checks passed!")
    click.echo("")


def check_first_run(store: MemoryStore) -> None:
    """Show first-run hint if DB was just created."""
    hint_shown = store._repo.load_metadata("onboarding_hint_shown")
    if hint_shown:
        return

    s = store.stats()
    if s.total_memories == 0:
        # Check metadata count -- if only embedder_state or nothing, it's fresh
        click.echo("Tip: Run 'np init' for guided setup, or just start storing memories.")
        store._repo.save_metadata("onboarding_hint_shown", "1")
