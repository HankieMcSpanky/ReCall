"""Generate NeuroPack architecture diagram as PNG using Pillow."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 2800, 3400
img = Image.new("RGB", (W, H), "#0a0a1a")
d = ImageDraw.Draw(img)

# Fonts
def font(size, bold=False):
    names = ["segoeuib.ttf", "segoeui.ttf", "arial.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except (OSError, IOError):
            pass
    try:
        return ImageFont.truetype("consola.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()

def mono(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except (OSError, IOError):
        return font(size)

F_TITLE = font(56, bold=True)
F_SUB = font(28)
F_SECTION = font(26, bold=True)
F_HEAD = font(28, bold=True)
F_BODY = font(22)
F_BODY_B = font(22, bold=True)
F_SMALL = font(20)
F_MONO = mono(22)
F_MONO_SM = mono(20)
F_BIG_NUM = font(48, bold=True)

# Colors
BG = "#0a0a1a"
RED = "#e94560"
GREEN = "#0ead69"
PURPLE = "#7b2ff7"
ORANGE = "#f77f00"
BLUE = "#88ddff"
WHITE = "#ffffff"
GRAY = "#8888aa"
DIM = "#555577"
TEXT = "#aaaacc"
DARK_PANEL = "#141428"

def rounded_rect(xy, fill, outline=None, r=16, width=2):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)

def box_with_header(x, y, w, h, header_text, header_color, body_color="#141428", outline_color=None, header_h=56):
    outline = outline_color or header_color
    rounded_rect((x, y, x+w, y+h), fill=body_color, outline=outline, r=16, width=2)
    rounded_rect((x, y, x+w, y+header_h), fill=header_color, r=16, width=0)
    d.rectangle((x+2, y+header_h-14, x+w-2, y+header_h), fill=body_color)
    d.text((x+24, y+12), header_text, fill=WHITE, font=F_HEAD)

def arrow_down(x, y1, y2, color=DIM, width=3):
    d.line([(x, y1), (x, y2-10)], fill=color, width=width)
    d.polygon([(x-8, y2-12), (x+8, y2-12), (x, y2)], fill=color)

def arrow_right(x1, y, x2, color=DIM, width=2):
    d.line([(x1, y), (x2-10, y)], fill=color, width=width)
    d.polygon([(x2-12, y-6), (x2-12, y+6), (x2, y)], fill=color)

# ===== TITLE =====
d.text((W//2, 60), "NeuroPack", fill=WHITE, font=F_TITLE, anchor="mt")
d.text((W//2, 130), "AI Memory Layer  --  Local, private memory for AI agents", fill=GRAY, font=F_SUB, anchor="mt")
d.text((W//2, 165), "with 3-level compression & shared learning", fill=GRAY, font=F_SUB, anchor="mt")

# ===== INTERFACES =====
d.text((80, 230), "INTERFACES", fill=RED, font=F_SECTION)
Y_IF = 270

# CLI
box_with_header(80, Y_IF, 540, 310, "CLI  (np)", "#2a1030", DARK_PANEL, RED)
cli_lines = [
    (TEXT, "np store    np recall   np list"),
    (TEXT, "np forget   np stats    np inspect"),
    (BLUE, "np init     np doctor"),
    ("#bb88ff", "np llm add/remove/list/test"),
    ("#ffaa44", "np agent create/log/learn/..."),
    (DIM, "np import/export/obsidian"),
]
for i, (col, txt) in enumerate(cli_lines):
    d.text((110, Y_IF+72+i*36), txt, fill=col, font=F_MONO_SM)

# REST API
box_with_header(680, Y_IF, 540, 310, "REST API  (:7341)", "#0a2a15", DARK_PANEL, GREEN)
api_lines = [
    (TEXT, "POST /v1/memories"),
    (TEXT, "POST /v1/recall"),
    (TEXT, "GET  /v1/stats"),
    ("#bb88ff", "GET  /v1/llms"),
    ("#ffaa44", "POST /v1/agents/{n}/log"),
    ("#ffaa44", "GET  /v1/agents/scoreboard"),
]
for i, (col, txt) in enumerate(api_lines):
    d.text((710, Y_IF+72+i*36), txt, fill=col, font=F_MONO_SM)

# MCP
box_with_header(1280, Y_IF, 540, 310, "MCP Server  (Claude)", "#0a1530", DARK_PANEL, "#4477cc")
mcp_lines = [
    (TEXT, "remember()  recall()"),
    (TEXT, "context_summary()"),
    (TEXT, "fetch_details()"),
    ("#bb88ff", "list_llms()  test_llm()"),
    ("#ffaa44", "agent_log()"),
    ("#ffaa44", "agent_scoreboard()"),
]
for i, (col, txt) in enumerate(mcp_lines):
    d.text((1310, Y_IF+72+i*36), txt, fill=col, font=F_MONO_SM)

# Extension/Desktop
box_with_header(1880, Y_IF, 540, 310, "Chrome Ext + Desktop", "#1a1808", DARK_PANEL, "#ddaa33")
ext_lines = [
    (TEXT, "Capture web pages,"),
    (TEXT, "highlights, and selections."),
    (TEXT, "Send to NeuroPack API."),
    ("", ""),
    (TEXT, "Desktop GUI via PyWebView"),
    (TEXT, "with web dashboard."),
]
for i, (col, txt) in enumerate(ext_lines):
    if col:
        d.text((1910, Y_IF+72+i*36), txt, fill=col, font=F_BODY)

# Arrows from interfaces to core
for x in [350, 950, 1550]:
    arrow_down(x, Y_IF+310, Y_IF+370, DIM, 3)

# ===== CORE ENGINE =====
Y_CORE = Y_IF + 370
rounded_rect((80, Y_CORE, 2500, Y_CORE+140), fill="#0f1a30", outline="#2a4a7a", r=16, width=2)
d.text((W//2-200, Y_CORE+25), "MemoryStore  --  Core Engine", fill=WHITE, font=font(34, True))
d.text((W//2-200, Y_CORE+75), "Single facade:  Privacy  ->  Compression  ->  Embedding  ->  Dedup  ->  Storage  ->  Knowledge Graph", fill="#8899bb", font=F_BODY)

# Arrows from core to subsystems
arrow_down(400, Y_CORE+140, Y_CORE+200, RED, 3)
arrow_down(1150, Y_CORE+140, Y_CORE+200, GREEN, 3)
arrow_down(1900, Y_CORE+140, Y_CORE+200, PURPLE, 3)

# ===== COMPRESSION =====
Y_SUB = Y_CORE + 200
box_with_header(80, Y_SUB, 660, 520, "3-Level Compression", RED, "#1a0a15", RED)

d.text((120, Y_SUB+72), "Input Text", fill=RED, font=F_BODY_B)
d.text((280, Y_SUB+72), "(e.g. 5000 tokens)", fill=DIM, font=F_SMALL)

# L3
rounded_rect((120, Y_SUB+105, 700, Y_SUB+195), fill="#221122", outline=RED, r=8, width=1)
d.text((140, Y_SUB+115), "L3 Abstract", fill="#ff6680", font=F_BODY_B)
d.text((340, Y_SUB+115), "~20 tokens", fill=DIM, font=F_SMALL)
d.text((140, Y_SUB+155), '"BTC dropped 5% after Fed rate hold"', fill=TEXT, font=F_MONO_SM)

# L2
rounded_rect((120, Y_SUB+210, 700, Y_SUB+330), fill="#221122", outline=RED, r=8, width=1)
d.text((140, Y_SUB+220), "L2 Key Facts", fill="#ff6680", font=F_BODY_B)
d.text((340, Y_SUB+220), "~50 tokens, 3-5 facts", fill=DIM, font=F_SMALL)
d.text((140, Y_SUB+260), '["BTC fell from 68k to 64.5k",', fill=TEXT, font=F_MONO_SM)
d.text((140, Y_SUB+288), ' "Fed held rates at 5.25-5.5%",', fill=TEXT, font=F_MONO_SM)

# L1
rounded_rect((120, Y_SUB+345, 700, Y_SUB+425), fill="#221122", outline=RED, r=8, width=1)
d.text((140, Y_SUB+355), "L1 Full Text", fill="#ff6680", font=F_BODY_B)
d.text((340, Y_SUB+355), "zstd compressed, lossless", fill=DIM, font=F_SMALL)
d.text((140, Y_SUB+395), "[binary blob -- fully recoverable]", fill=TEXT, font=F_MONO_SM)

d.text((410, Y_SUB+465), "Token savings: ~90% (L3) / ~80% (L2)", fill=DIM, font=F_SMALL, anchor="mt")

# ===== HYBRID SEARCH =====
box_with_header(810, Y_SUB, 660, 520, "Hybrid Search", GREEN, "#0a1a10", GREEN)

d.text((850, Y_SUB+72), "Query:", fill=GREEN, font=F_BODY_B)
d.text((940, Y_SUB+72), '"What happened with BTC?"', fill=DIM, font=F_SMALL)

# Vector
rounded_rect((850, Y_SUB+105, 1430, Y_SUB+210), fill="#112218", outline=GREEN, r=8, width=1)
d.text((870, Y_SUB+118), "Vector Search", fill="#44dd88", font=F_BODY_B)
d.text((870, Y_SUB+155), "TF-IDF embeddings (256d feature hash)", fill=TEXT, font=F_BODY)
d.text((870, Y_SUB+183), "Cosine similarity scoring", fill=TEXT, font=F_BODY)
# Weight badge
rounded_rect((1350, Y_SUB+118, 1420, Y_SUB+148), fill=GREEN, r=6, width=0)
d.text((1385, Y_SUB+126), "0.6", fill=WHITE, font=F_BODY_B, anchor="mt")

# FTS
rounded_rect((850, Y_SUB+225, 1430, Y_SUB+330), fill="#112218", outline=GREEN, r=8, width=1)
d.text((870, Y_SUB+238), "Full-Text Search", fill="#44dd88", font=F_BODY_B)
d.text((870, Y_SUB+275), "SQLite FTS5, BM25 ranking", fill=TEXT, font=F_BODY)
d.text((870, Y_SUB+303), "Porter + Unicode tokenizer", fill=TEXT, font=F_BODY)
# Weight badge
rounded_rect((1350, Y_SUB+238, 1420, Y_SUB+268), fill=GREEN, r=6, width=0)
d.text((1385, Y_SUB+246), "0.4", fill=WHITE, font=F_BODY_B, anchor="mt")

# Fusion
arrow_down(1140, Y_SUB+335, Y_SUB+370, GREEN, 3)
rounded_rect((900, Y_SUB+370, 1380, Y_SUB+420), fill=GREEN, r=8, width=0)
d.text((1140, Y_SUB+385), "Reciprocal Rank Fusion (k=60)", fill=WHITE, font=F_BODY_B, anchor="mt")

d.text((1140, Y_SUB+465), "Weighted merge of both search signals", fill=DIM, font=F_SMALL, anchor="mt")

# ===== LLM REGISTRY =====
box_with_header(1540, Y_SUB, 660, 520, "LLM Registry", PURPLE, "#120a22", PURPLE)

d.text((1580, Y_SUB+72), "Named multi-provider configs in SQLite", fill="#bb99ee", font=F_BODY)

# Table
table_x, table_y = 1580, Y_SUB+105
rounded_rect((table_x, table_y, table_x+580, table_y+38), fill="#2a1a44", r=6, width=0)
d.text((table_x+20, table_y+8), "Name", fill=PURPLE, font=F_BODY_B)
d.text((table_x+200, table_y+8), "Provider", fill=PURPLE, font=F_BODY_B)
d.text((table_x+400, table_y+8), "Model", fill=PURPLE, font=F_BODY_B)

rows = [
    ("local-llama", "openai-compat", "llama3.2"),
    ("gpt4", "openai", "gpt-4o-mini"),
    ("claude", "anthropic", "haiku"),
]
for i, (name, prov, model) in enumerate(rows):
    ry = table_y + 40 + i*34
    bg = "#1a1030" if i % 2 == 0 else "#150d28"
    d.rectangle((table_x, ry, table_x+580, ry+34), fill=bg)
    d.text((table_x+20, ry+6), name, fill=TEXT, font=F_MONO_SM)
    d.text((table_x+200, ry+6), prov, fill=TEXT, font=F_MONO_SM)
    d.text((table_x+400, ry+6), model, fill=TEXT, font=F_MONO_SM)

d.text((1580, Y_SUB+270), "Supports any OpenAI-compatible endpoint:", fill="#8866bb", font=F_BODY_B)
d.text((1580, Y_SUB+305), "Ollama, vLLM, Together, Groq, Azure, LM Studio", fill=DIM, font=F_BODY)
d.text((1580, Y_SUB+345), "Default LLM powers smarter L3/L2 compression", fill=DIM, font=F_BODY)
d.text((1580, Y_SUB+380), "Legacy env var config auto-migrates on first run", fill=DIM, font=F_BODY)

# Dashed arrow from LLM to compression
d.text((1580, Y_SUB+430), "powers compression  -->", fill="#7b2ff7", font=F_SMALL)

# ===== STORAGE =====
Y_STOR = Y_SUB + 560
d.text((80, Y_STOR), "STORAGE", fill="#3388cc", font=F_SECTION)
Y_STOR += 40
rounded_rect((80, Y_STOR, 2500, Y_STOR+180), fill="#0a1525", outline="#2a5580", r=16, width=2)

tables = [
    (110, "memories", ["id, content, l3, l2, l1", "embedding, tags, namespace", "priority, tokens, timestamps"], "#55aaee"),
    (600, "memories_fts", ["FTS5 virtual table", "Auto-synced via triggers", "Porter + Unicode tokenizer"], "#55aaee"),
    (1080, "entities", ["Knowledge graph nodes", "name, type, namespace", "mention_count, timestamps"], "#55aaee"),
    (1560, "relationships", ["Knowledge graph edges", "source, target, relation", "weight, memory_id"], "#55aaee"),
    (2020, "metadata", ["Key/value store", "LLM registry, agent configs", "embedder state, flags"], "#55aaee"),
]
for tx, name, lines, col in tables:
    rounded_rect((tx, Y_STOR+18, tx+440, Y_STOR+160), fill="#112240", outline="#3388cc", r=8, width=1)
    d.text((tx+20, Y_STOR+28), name, fill=col, font=F_BODY_B)
    for i, line in enumerate(lines):
        d.text((tx+20, Y_STOR+62+i*30), line, fill="#8899bb", font=F_MONO_SM)

# ===== MULTI-AGENT LEARNING =====
Y_AG = Y_STOR + 220
d.text((80, Y_AG), "MULTI-AGENT LEARNING", fill=ORANGE, font=F_SECTION)
d.text((520, Y_AG), 'The "Moltbook for AIs" -- competing agents share a collective memory', fill="#886633", font=F_BODY)

Y_AG += 45
rounded_rect((80, Y_AG, 2500, Y_AG+680), fill="#1a1208", outline=ORANGE, r=16, width=1)

# Agent boxes
agents = [
    (120, "trader1", [
        ("#ff6666", '"Lost 2% on bad entry"'),
        ("#ff6666", "  auto-tag: [mistake]"),
        ("#66ff88", '"Gained 5% on breakout"'),
        ("#66ff88", "  auto-tag: [win]"),
    ]),
    (620, "trader2", [
        ("#66ff88", '"Profit on momentum play"'),
        ("#66ff88", "  auto-tag: [win]"),
        ("#66ff88", '"Won big on ETH long"'),
        ("#66ff88", "  auto-tag: [win]"),
    ]),
    (1120, "trader3", [
        ("#ff6666", '"Error in sizing calc"'),
        ("#ff6666", "  auto-tag: [mistake]"),
        (TEXT, '"Market consolidating"'),
        (TEXT, "  auto-tag: [observation]"),
    ]),
]

for ax, aname, alines in agents:
    box_with_header(ax, Y_AG+25, 440, 260, aname, ORANGE, "#1a1510", ORANGE, header_h=50)
    d.text((ax+30, Y_AG+90), f"namespace: {aname}", fill=DIM, font=F_SMALL)
    for i, (col, txt) in enumerate(alines):
        d.text((ax+30, Y_AG+125+i*32), txt, fill=col, font=F_MONO_SM)

# Share arrows
for ax in [340, 840, 1340]:
    arrow_down(ax, Y_AG+290, Y_AG+340, ORANGE, 3)
    d.text((ax+15, Y_AG+305), "share", fill=ORANGE, font=F_SMALL)

# Shared namespace
rounded_rect((120, Y_AG+340, 1520, Y_AG+460), fill="#2a1a00", outline=ORANGE, r=12, width=3)
d.text((820, Y_AG+365), "shared  (namespace)", fill=ORANGE, font=font(30, True), anchor="mt")
d.text((820, Y_AG+410), "Collective memory pool -- all agents' shared learnings accumulate here", fill="#ccaa66", font=F_BODY, anchor="mt")

# Learn arrow
arrow_down(820, Y_AG+460, Y_AG+510, ORANGE, 3)
d.text((845, Y_AG+478), "agent learn", fill=ORANGE, font=F_SMALL)

# Scoreboard
box_with_header(1620, Y_AG+25, 840, 640, "Scoreboard", ORANGE, "#1a1510", ORANGE, header_h=50)

# Scoreboard table
sb_x = 1660
sb_y = Y_AG + 100
rounded_rect((sb_x, sb_y, sb_x+760, sb_y+38), fill="#2a1a00", r=6, width=0)
for j, col_name in enumerate(["Agent", "W", "L", "Ratio", "Total"]):
    d.text((sb_x+20+j*150, sb_y+8), col_name, fill=ORANGE, font=F_BODY_B)

sb_data = [
    ("#66ff88", ["trader2", "2", "0", "1.00", "2"]),
    ("#ffaa44", ["trader1", "1", "1", "0.50", "2"]),
    ("#ff6666", ["trader3", "0", "1", "0.00", "2"]),
]
for i, (col, vals) in enumerate(sb_data):
    ry = sb_y + 42 + i*36
    for j, v in enumerate(vals):
        d.text((sb_x+20+j*150, ry), v, fill=col, font=F_MONO)

# Auto-tag legend
lg_y = sb_y + 180
d.text((sb_x, lg_y), "Auto-tagging keywords:", fill="#886633", font=F_BODY_B)
legend = [
    ("#1a3a1a", "#66ff88", "win", "profit, gained, success, won"),
    ("#3a1a1a", "#ff6666", "mistake", "loss, error, failed, wrong"),
    ("#1a1a2a", TEXT, "obs", "everything else"),
]
for i, (bg, fg, label, desc) in enumerate(legend):
    ly = lg_y + 40 + i*40
    rounded_rect((sb_x, ly, sb_x+100, ly+30), fill=bg, r=6, width=0)
    d.text((sb_x+50, ly+4), label, fill=fg, font=F_BODY_B, anchor="mt")
    d.text((sb_x+120, ly+4), desc, fill=DIM, font=F_SMALL)

# Agent commands
cmd_y = lg_y + 180
d.text((sb_x, cmd_y), "Commands:", fill="#886633", font=F_BODY_B)
cmds = [
    "np agent create <name>",
    "np agent log <name> <text>",
    "np agent mistakes / wins <name>",
    "np agent share <name> <id>",
    "np agent learn <name>",
    "np agent scoreboard",
]
for i, cmd in enumerate(cmds):
    d.text((sb_x, cmd_y+30+i*28), cmd, fill=TEXT, font=F_MONO_SM)

# ===== ONBOARDING =====
Y_ON = Y_AG + 720
d.text((80, Y_ON), "ONBOARDING", fill=BLUE, font=F_SECTION)
Y_ON += 40

# np init
box_with_header(80, Y_ON, 780, 290, "np init  --  Setup Wizard", "#1a3050", "#0a1520", BLUE)
init_lines = [
    "[1/4] Database ............ create ~/.neuropack/",
    "[2/4] LLM Config .......... provider, key, model",
    "[3/4] API Security ........ auth token",
    "[4/4] Example Memories .... 3 starter memories",
]
for i, line in enumerate(init_lines):
    d.text((120, Y_ON+72+i*36), line, fill=TEXT, font=F_MONO_SM)
d.text((120, Y_ON+240), "Writes config.env, tests LLM connection, shows next steps", fill=DIM, font=F_SMALL)

# np doctor
box_with_header(920, Y_ON, 780, 290, "np doctor  --  Health Check", "#1a3050", "#0a1520", BLUE)
doc_lines = [
    ("#66ff88", "Database .............. OK  (45 KB)"),
    ("#66ff88", "Schema ................ OK  (v3)"),
    ("#66ff88", "Memory count .......... 42 memories"),
    ("#66ff88", "LLM (local-llama) .... OK  (0.2s)"),
    ("#ff6666", "LLM (gpt4-backup) .... FAIL (401)"),
]
for i, (col, line) in enumerate(doc_lines):
    d.text((960, Y_ON+72+i*36), line, fill=col, font=F_MONO_SM)
d.text((960, Y_ON+252), "Tests all LLM connections, reports issues", fill=DIM, font=F_SMALL)

# Data flow
box_with_header(1760, Y_ON, 720, 290, "Data Flow Summary", "#1a2540", "#0a1520", "#556688")
flows = [
    (RED, "Store:", "text -> privacy -> compress -> embed -> dedup -> DB"),
    (GREEN, "Recall:", "query -> embed -> vector + FTS5 -> rank fusion"),
    (ORANGE, "Agent:", "log -> auto-tag -> namespace -> share -> learn"),
    (PURPLE, "LLM:", "registry -> provider dispatch -> powers compression"),
]
for i, (col, label, desc) in enumerate(flows):
    d.text((1800, Y_ON+80+i*45), label, fill=col, font=F_BODY_B)
    d.text((1910, Y_ON+80+i*45), desc, fill=TEXT, font=F_SMALL)

# ===== FOOTER =====
Y_FT = Y_ON + 360
rounded_rect((80, Y_FT, 2500, Y_FT+130), fill="#0f0f20", outline="#222244", r=16, width=1)

stats_data = [
    (200, RED, "202", "tests pass"),
    (530, GREEN, "14", "new CLI commands"),
    (860, PURPLE, "4", "new MCP tools"),
    (1190, ORANGE, "4", "new API endpoints"),
    (1520, BLUE, "9", "new source files"),
    (1850, "#ddaa33", "6", "modified files"),
]
for sx, col, num, label in stats_data:
    d.text((sx, Y_FT+25), num, fill=col, font=F_BIG_NUM)
    d.text((sx, Y_FT+85), label, fill=DIM, font=F_BODY)

d.text((W//2, Y_FT+145), "NeuroPack v3  --  Enterprise AI Memory Layer  --  Local, Private, Zero Cloud Dependencies",
       fill="#333355", font=F_BODY, anchor="mt")

# Save
out_path = r"C:\dev\neuropack\architecture.png"
img.save(out_path, "PNG", optimize=True)
print(f"Saved to {out_path} ({os.path.getsize(out_path) // 1024} KB)")
