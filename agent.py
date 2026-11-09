#!/usr/bin/env python3
"""
Agent CLI — Multi-model local agent with Gemini web search.
Inspired by Claude CLI: REPL loop, history, routing, bash execution, aider integration.
"""

import gzip
import json
import os
import re
import readline
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────────────
theme = Theme({"info": "dim cyan", "warn": "yellow", "error": "bold red", "ok": "bold green"})
console = Console(theme=theme)

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    "brain":     "gurubot/gpt-oss-derestricted:20b",
    "reasoning": "huihui_ai/deepseek-r1-abliterated:7b",
    "coding":    "huihui_ai/qwen2.5-abliterate:7b-instruct",
    "security":  "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B:latest",
    "chat":      "dolphin-llama3:8b",
    "fast":      "goekdenizguelmez/JOSIEFIED-Qwen3:1.7b",
    "light":     "nidumai/nidum-gemma-3-4b-it-uncensored:q3_k_m",
    "manager":   "deepseek-r1:1.5b",
}

GEMINI_MODEL = "gemini-2.5-pro"

CATEGORIES = ["reasoning", "coding", "security", "chat", "fast", "light", "search", "bash"]

SEARCH_TRIGGERS = (
    "look up online", "search for", "find online", "latest version",
    "current price", "what's the news", "today's", "right now", "as of today",
)

BASH_TRIGGERS = (
    "laptop specs", "system specs", "hardware specs", "my specs",
    "cpu info", "ram info", "memory info", "disk info", "storage info",
    "what cpu", "what gpu", "what ram", "how much ram", "how much disk",
    "system info", "os info", "os version", "kernel version",
    "show specs", "check specs", "get specs", "look up specs",
    "what is my", "what's my",
)

# Keyword → command fallbacks for when the model fails to output a bash block
_BASH_FALLBACK_MAP = (
    ({"spec", "specs", "hardware", "components"},
     "lscpu | grep -E 'Model name|CPU\\(s\\)|Thread|MHz' && echo && free -h && echo && df -h / && echo && lspci | grep -Ei 'vga|3d|display' && echo && uname -srm && (lsb_release -d 2>/dev/null || grep PRETTY /etc/os-release)"),
    ({"cpu", "processor"},        "lscpu"),
    ({"ram", "memory"},           "free -h"),
    ({"disk", "storage", "space"},"df -h"),
    ({"gpu", "graphics"},         "lspci | grep -Ei 'vga|3d|display'"),
    ({"os", "kernel", "version"}, "uname -srm && (lsb_release -a 2>/dev/null || cat /etc/os-release)"),
    ({"process", "processes"},    "ps aux --sort=-%cpu | head -15"),
    ({"temperature", "temp"},     "sensors 2>/dev/null || paste /sys/class/thermal/thermal_zone*/temp"),
    ({"network", "ip", "wifi", "lan", "wlan", "ethernet", "connected", "connection", "ssid"},
     "ip route show default && echo && ip addr show && echo && nmcli device status 2>/dev/null && echo && iwconfig 2>/dev/null | grep -E 'ESSID|Signal|Access'"),
)

def bash_fallback(task: str):
    words = set(task.lower().split())
    for keywords, cmd in _BASH_FALLBACK_MAP:
        if words & keywords:
            return cmd
    return None


BASH_SYSTEM = {
    "role": "system",
    "content": (
        "You are a shell assistant on Ubuntu Linux. "
        "Respond ONLY with a single ```bash code block. No prose, no explanation, nothing else.\n\n"
        "Example — user: 'show disk usage'\n"
        "Your response:\n"
        "```bash\n"
        "df -h\n"
        "```"
    ),
}

CODING_SYSTEM = {
    "role": "system",
    "content": (
        "You are a coding assistant. Respond with a single complete code block "
        "using the appropriate language fence (```python, ```bash, etc.). "
        "No explanation, no prose — just the code block."
    ),
}

COMPLEX_TRIGGERS = (
    "edit file", "modify", "refactor", "multiple files",
    "across files", "update file", "rewrite file",
)

# ── Persistent history ────────────────────────────────────────────────────────
HISTORY_FILE = Path.home() / ".local/share/agent/history.json"
HISTORY_ARCHIVE_DIR = Path.home() / ".local/share/agent/archives"
HISTORY_MAX = 50
HISTORY_SEED = 10

# Conversation history (persists across turns like Claude CLI)
history: list[dict] = []


def load_history() -> None:
    """Load history from disk on startup."""
    global history
    if not HISTORY_FILE.exists():
        return
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            history = data
            console.print(f"[info]Loaded {len(history)} messages from history.[/info]")
    except Exception as e:
        console.print(f"[warn]Could not load history: {e}[/warn]")


def save_history() -> None:
    """Persist current history to disk after each turn."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        console.print(f"[warn]Could not save history: {e}[/warn]")


def maybe_archive_history() -> None:
    """If history > HISTORY_MAX, gzip-archive and reset to last HISTORY_SEED messages."""
    if len(history) <= HISTORY_MAX:
        return
    try:
        HISTORY_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = HISTORY_ARCHIVE_DIR / f"history_{stamp}.json.gz"
        with gzip.open(archive_path, "wt", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Prune old archives — keep only 10 most recent
        archives = sorted(HISTORY_ARCHIVE_DIR.glob("history_*.json.gz"))
        for old in archives[:-10]:
            old.unlink()

        # Seed the new session with the last HISTORY_SEED messages
        seed = history[-HISTORY_SEED:]
        history.clear()
        history.extend(seed)
        console.print(f"[info]History archived to {archive_path.name}. Session reset with last {HISTORY_SEED} messages.[/info]")
        save_history()
    except Exception as e:
        console.print(f"[warn]Could not archive history: {e}[/warn]")


def clear_history() -> None:
    """Wipe in-memory history and delete the history file."""
    history.clear()
    try:
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
    except Exception as e:
        console.print(f"[warn]Could not delete history file: {e}[/warn]")
    console.print("[info]History cleared.[/info]")


# ── Ollama ───────────────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: list[dict]) -> str:
    """Call Ollama REST API with streaming output."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
        full = ""
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            full += token
            print(token, end="", flush=True)
            if chunk.get("done"):
                break
        print()
        return full
    except Exception as e:
        return f"[Ollama error] {e}"


def route(task: str) -> str:
    """Chat model classifies the task into a category."""
    prompt = (
        "You are a task classifier. Output exactly ONE word from this list:\n"
        "reasoning, coding, security, chat, fast, light, search, bash\n\n"
        "Rules (in priority order):\n"
        "1. 'search'   — needs current/live/web information (news, weather, prices, latest versions)\n"
        "2. 'bash'     — run, execute, show, check, list, monitor system commands or shell operations\n"
        "3. 'coding'   — write, create, build, fix, debug, refactor, implement any code or script in any language\n"
        "4. 'security' — network scanning, pentesting, firewall, vulnerability, encryption, auth analysis\n"
        "5. 'reasoning'— math, logic puzzles, step-by-step analysis, planning without code\n"
        "6. 'chat'     — casual conversation, opinions, general questions answerable from memory\n"
        "7. 'fast'     — simple one-line factual questions needing a quick answer\n"
        "8. 'light'    — trivial tasks, greetings, yes/no questions\n\n"
        "Examples:\n"
        "  'write a Python function to sort a list' -> coding\n"
        "  'create a bash script to backup files'  -> coding\n"
        "  'fix this bug in my JavaScript'         -> coding\n"
        "  'what is the weather today?'            -> search\n"
        "  'look up online latest Python release'  -> search\n"
        "  'current price of RTX 4090'             -> search\n"
        "  'show disk usage'                       -> bash\n"
        "  'look up the specs of this laptop'      -> bash\n"
        "  'show system info'                      -> bash\n"
        "  'what CPU do I have'                    -> bash\n"
        "  'check RAM usage'                       -> bash\n"
        "  'list running processes'                -> bash\n"
        "  'scan for open ports'                   -> security\n"
        "  'what is 12 * 34?'                      -> fast\n"
        "  'explain recursion step by step'        -> reasoning\n"
        "  'how are you?'                          -> chat\n\n"
        f"Task: {task}\n\n"
        "Reply with one word only, no punctuation."
    )
    console.print("[info]Routing...[/info]")
    result = subprocess.run(
        ["ollama", "run", MODELS["chat"], prompt],  # 8b instruct model for reliable classification
        capture_output=True, text=True, timeout=120,
    )
    raw = result.stdout.strip().lower()
    # Strip <think>...</think> blocks produced by reasoning models
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Check the last non-empty line first (where the final answer lands)
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if line in CATEGORIES:
            return line
    # Fallback: scan full output for any category word
    for cat in CATEGORIES:
        if cat in raw:
            return cat
    return "chat"


# ── Gemini search ─────────────────────────────────────────────────────────────

def gemini_search(query: str) -> str:
    """Run Gemini 2.0 Flash in headless mode for web-grounded answers."""
    console.print(f"[info]Gemini {GEMINI_MODEL} searching...[/info]")
    result = subprocess.run(
        ["gemini", "-m", GEMINI_MODEL, "-p", query],
        capture_output=True, text=True, timeout=90,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    err = result.stderr.strip()
    return f"[Search error] {err or 'No output returned'}"


# ── Bash execution ────────────────────────────────────────────────────────────

def run_bash(cmd: str) -> str:
    """Show command, ask for confirmation, then execute."""
    console.print(Panel(f"[warn]{cmd}[/warn]", title="[error]AI wants to run[/error]", expand=False))
    confirm = Prompt.ask("Allow?", choices=["y", "n"], default="n")
    if confirm != "y":
        return "Command aborted by user."
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    return f"STDOUT:\n{out}\nSTDERR:\n{err}" if err else out


# ── Aider ─────────────────────────────────────────────────────────────────────

def launch_aider(model: str, task: str) -> None:
    """Hand off coding/security/reasoning tasks to aider."""
    console.print(f"[info]Launching Aider → {model}[/info]")
    instruction = (
        f"{task}. "
        "If packages need installing, include the install command. "
        "Implement directly — do not ask me to run it."
    )
    subprocess.run(
        ["aider", "--model", f"ollama_chat/{model}", "--message", instruction, "--yes-always"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )


# ── Specs parser (100% accurate, no model involved) ──────────────────────────

def parse_specs(output: str) -> str | None:
    """Parse the known specs command output into a formatted spec sheet.
    Returns None if the output doesn't contain recognisable spec fields."""
    cpu_model  = re.search(r"Model name:\s+(.+)",           output)
    cpu_total  = re.search(r"^CPU\(s\):\s+(\d+)",           output, re.MULTILINE)
    threads_pc = re.search(r"Thread\(s\) per core:\s+(\d+)", output)
    cpu_max    = re.search(r"CPU max MHz:\s+([\d.]+)",        output)
    ram_line   = re.search(r"Mem:\s+(\S+)\s+(\S+)\s+(\S+)", output)
    disk_line  = re.search(r"(/dev/\S+)\s+(\S+)\s+\S+\s+\S+\s+(\d+%)\s+/", output)
    gpu_line   = re.search(r"VGA compatible controller: (.+)", output)
    kernel     = re.search(r"(Linux \S+ \S+)",               output)
    os_desc    = re.search(r"Description:\s+(.+)",           output)

    if not cpu_model:
        return None  # doesn't look like a specs dump

    lines = []

    # CPU — compute cores correctly from total threads ÷ threads-per-core
    cpu_str = cpu_model.group(1).strip()
    if cpu_total and threads_pc:
        total   = int(cpu_total.group(1))
        tpc     = int(threads_pc.group(1))
        cores   = total // tpc
        cpu_str += f"  ({cores} cores / {total} threads"
        if cpu_max:
            ghz = float(cpu_max.group(1)) / 1000
            cpu_str += f", up to {ghz:.2f} GHz"
        cpu_str += ")"
    lines.append(f"**CPU:** {cpu_str}")

    if ram_line:
        lines.append(f"**RAM:** {ram_line.group(1)} total  |  {ram_line.group(2)} used  |  {ram_line.group(3)} free")

    if disk_line:
        lines.append(f"**Disk:** {disk_line.group(1)}  {disk_line.group(2)} total  ({disk_line.group(3)} used)")

    if gpu_line:
        lines.append(f"**GPU:** {gpu_line.group(1).strip()}")

    if kernel:
        lines.append(f"**Kernel:** {kernel.group(1)}")

    if os_desc:
        lines.append(f"**OS:** {os_desc.group(1).strip()}")

    return "\n".join(lines)


# ── Auto file injection ───────────────────────────────────────────────────────

def auto_inject_files(task: str) -> str:
    """Find all file paths in task, prepend their contents if they exist."""
    # Match absolute paths, ~/paths, ./paths, and quoted paths
    pattern = r'(?:"(/[^"]+)"|\'(/[^\']+)\'|(~/[^\s\'"]+)|(\.\/[^\s\'"]+)|(/[^\s\'"]+))'
    matches = re.findall(pattern, task)
    injected = []
    for groups in matches:
        raw = next((g for g in groups if g), "")
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_file():
            try:
                content = path.read_text(errors="replace")
                injected.append(f"Contents of {path}:\n```\n{content}\n```")
            except Exception:
                pass
    if not injected:
        return task
    prefix = "\n\n".join(injected)
    console.print(f"[info]Auto-injected {len(injected)} file(s).[/info]")
    return f"{prefix}\n\nUser question: {task}"


# ── Inline execution ──────────────────────────────────────────────────────────

def ask_execution_mode() -> str:
    """Show menu and return 'inline', 'aider', or 'skip'."""
    console.print(Panel(
        "[bold]How do you want to run this task?[/bold]\n\n"
        "  [cyan][1][/cyan] Inline   — run code block here\n"
        "  [cyan][2][/cyan] Aider    — full editor session\n"
        "  [cyan][3][/cyan] Skip     — cancel",
        title="Execution mode",
        expand=False,
    ))
    choice = Prompt.ask("Choice", choices=["1", "2", "3"], default="1")
    return {"1": "inline", "2": "aider", "3": "skip"}[choice]


def run_inline(category: str, task: str) -> None:
    """Get code from model, show it, confirm, run it, analyse output."""
    model = MODELS[category]
    msgs = [CODING_SYSTEM, {"role": "user", "content": task}]
    console.print(f"[info]Generating code with {model}...[/info]")
    response = ollama_chat(model, msgs)

    # Extract first code block (any language)
    match = re.search(r"```(\w+)?\n(.*?)\n```", response, re.DOTALL)
    if not match:
        console.print(Panel(Markdown(response), title="[ok]Response (no runnable block found)[/ok]"))
        history.append({"role": "assistant", "content": response})
        return

    lang = (match.group(1) or "").lower()
    code = match.group(2)

    console.print(Panel(Markdown(f"```{lang}\n{code}\n```"), title=f"[ok]Generated {lang or 'code'}[/ok]"))
    confirm = Prompt.ask("Run this?", choices=["y", "n"], default="n")
    if confirm != "y":
        history.append({"role": "assistant", "content": response})
        return

    if lang in ("python", "py", "python3"):
        proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
    elif lang in ("bash", "sh", "shell"):
        proc = subprocess.run(code, shell=True, capture_output=True, text=True)
    else:
        console.print(f"[warn]Don't know how to run '{lang}' inline — use Aider instead.[/warn]")
        history.append({"role": "assistant", "content": response})
        return

    out = proc.stdout.strip()
    err = proc.stderr.strip()
    output_text = f"STDOUT:\n{out}\nSTDERR:\n{err}" if err else (out or "(no output)")
    console.print(Panel(output_text, title="Output"))

    history.append({"role": "assistant", "content": response})
    history.append({"role": "system", "content": f"Code output:\n{output_text}"})

    # Let the chat model analyse the result
    analysis = ollama_chat(MODELS["chat"], history + [{"role": "user", "content": "Analyse the output above."}])
    console.print(Panel(Markdown(analysis), title="[ok]Analysis[/ok]"))
    history.append({"role": "assistant", "content": analysis})


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process(task: str) -> None:
    # ── Keyword pre-check: bypass router for known local/web patterns ─────
    task_lower = task.lower()
    if any(t in task_lower for t in BASH_TRIGGERS):
        category = "bash"
    elif any(t in task_lower for t in SEARCH_TRIGGERS):
        category = "search"
    else:
        category = route(task)
    console.print(f"[ok]→ {category.upper()}[/ok]")

    # ── Web search via Gemini ──────────────────────────────────────────────
    if category == "search":
        answer = gemini_search(task)
        console.print(Panel(Markdown(answer), title=f"[ok]Gemini {GEMINI_MODEL}[/ok]"))
        history.append({"role": "assistant", "content": answer})
        save_history()
        maybe_archive_history()
        return

    # ── Code / security / reasoning → menu or auto-Aider ──────────────────
    if category in ("coding", "security", "reasoning"):
        is_complex = any(t in task_lower for t in COMPLEX_TRIGGERS)
        if is_complex:
            console.print("[info]Complex task detected — launching Aider directly.[/info]")
            launch_aider(MODELS[category], task)
        else:
            mode = ask_execution_mode()
            if mode == "inline":
                run_inline(category, task)
            elif mode == "aider":
                launch_aider(MODELS[category], task)
            else:
                console.print("[info]Skipped.[/info]")
        save_history()
        maybe_archive_history()
        return

    # ── Bash tasks: force model to output a command block ─────────────────
    if category == "bash":
        # For known query patterns, use the curated fallback directly — skips
        # the model and guarantees clean, concise output every time.
        cmd = bash_fallback(task)
        if cmd:
            console.print("[info]Using built-in command.[/info]")
        else:
            model = MODELS["coding"]  # 7b instruct model follows formatting better than fast (1.7b)
            msgs = [BASH_SYSTEM, {"role": "user", "content": task}]
            response = ollama_chat(model, msgs)
            console.print(Panel(Markdown(response), title="[ok]Agent (bash)[/ok]"))
            match = re.search(r"```bash\n(.*?)\n```", response, re.DOTALL)
            if match:
                cmd = match.group(1)
            else:
                # Model skipped fences — use response if it looks like a shell command
                stripped = response.strip()
                bare_cmd = (
                    stripped
                    if stripped and re.match(r"^[\w/\.\-]", stripped)
                       and any(k in stripped for k in ("|", "&&", ";", "lscpu", "free", "df", "lspci", "uname", "ps", "ip", "cat", "ls", "echo"))
                    else None
                )
                cmd = bare_cmd
                if not cmd:
                    console.print("[warn]Model didn't output a usable command.[/warn]")
        if cmd:
            result = run_bash(cmd)
            console.print(Panel(result, title="Command output"))
            stored = result[:1500] + "\n...(truncated)" if len(result) > 1500 else result
            history.append({"role": "system", "content": f"Command output:\n{stored}"})
            # Try the deterministic parser first; fall back to model for unknown output
            analysis = parse_specs(result)
            if analysis:
                console.print(Panel(Markdown(analysis), title="[ok]Specs[/ok]"))
            else:
                analysis_msgs = [
                    {"role": "system", "content": "You are a helpful assistant. Summarise the following command output in plain English. Focus on key facts (model names, sizes, versions). Be concise."},
                    {"role": "user", "content": f"Command output:\n{stored}\n\nSummarise the key information above."},
                ]
                analysis = ollama_chat(MODELS["fast"], analysis_msgs)
                console.print(Panel(Markdown(analysis), title="[ok]Analysis[/ok]"))
            history.append({"role": "assistant", "content": analysis})
        save_history()
        maybe_archive_history()
        return

    # ── Everything else → Ollama chat with feedback loop ─────────────────
    model = MODELS.get(category, MODELS["chat"])
    history.append({"role": "user", "content": task})

    response = ollama_chat(model, history)
    console.print(Panel(Markdown(response), title=f"[ok]Agent ({category})[/ok]"))
    history.append({"role": "assistant", "content": response})

    # If the response contains a bash block, offer to run it
    match = re.search(r"```bash\n(.*?)\n```", response, re.DOTALL)
    if match:
        result = run_bash(match.group(1))
        console.print(Panel(result, title="Command output"))
        history.append({"role": "system", "content": f"Command output:\n{result}"})

        # Feedback loop: let the model analyse the result
        analysis = ollama_chat(model, history)
        console.print(Panel(Markdown(analysis), title="[ok]Analysis[/ok]"))
        history.append({"role": "assistant", "content": analysis})

    save_history()
    maybe_archive_history()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    load_history()

    console.print(Panel(
        "[bold cyan]Agent CLI[/bold cyan]\n"
        "[info]Multi-model routing  •  Gemini 2.0 Flash search  •  Bash execution  •  Aider integration[/info]\n"
        "[info]Commands: exit · quit · clear (reset history)[/info]",
        title="[bold]Welcome[/bold]",
    ))

    # Single-shot mode (like `claude -p "..."`)
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        task = auto_inject_files(task)
        console.print(f"[bold]Task:[/bold] {task}")
        process(task)
        return

    # Interactive REPL
    while True:
        try:
            console.print(Rule(style="dim"))
            task = Prompt.ask("[bold cyan]>[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            save_history()
            console.print("\n[info]Goodbye.[/info]")
            break

        task = task.strip()
        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            save_history()
            console.print("[info]Goodbye.[/info]")
            break
        if task.lower() == "clear":
            clear_history()
            continue

        task = auto_inject_files(task)
        process(task)


if __name__ == "__main__":
    main()
