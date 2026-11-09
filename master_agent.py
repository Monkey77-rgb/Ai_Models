import subprocess
import sys

# Your complete arsenal
AGENT_REGISTRY = {
    "brain": "gurubot/gpt-oss-derestricted:20b",
    "reasoning": "huihui_ai/deepseek-r1-abliterated:7b",
    "coding": "huihui_ai/qwen2.5-abliterate:7b-instruct",
    "security": "WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-8B:latest",
    "chat": "dolphin-llama3:8b",
    "fast": "goekdenizguelmez/JOSIEFIED-Qwen3:1.7b",
    "light": "nidumai/nidum-gemma-3-4b-it-uncensored:q3_k_m"
}

VALID_CATEGORIES = ["reasoning", "coding", "security", "chat", "fast", "light"]


def call_brain(task):
    """Ask the brain model to classify the task into a specialist category."""
    prompt = (
        f"Classify this task into exactly ONE of these categories: "
        f"reasoning, coding, security, chat, fast, light. "
        f"Task: {task}\n"
        f"Reply with only the single category name, nothing else."
    )
    print(f"\n[Master] Brain routing task...")
    result = subprocess.run(
        ["ollama", "run", AGENT_REGISTRY["brain"], prompt],
        capture_output=True,
        text=True,
        timeout=120
    )
    raw = result.stdout.strip().lower()
    for category in VALID_CATEGORIES:
        if category in raw:
            return category
    # Fallback: default to coding if brain response is ambiguous
    print(f"[Master] Brain returned ambiguous response: '{raw}' — defaulting to 'coding'")
    return "coding"


def launch_aider(model, task):
    print(f"\n[Master] Launching Aider Agent with {model}...")
    instruction = (
        f"{task}. "
        "If you need to install a package, include the install command in your solution. "
        "Implement it directly — do not ask me to run it for you."
    )
    subprocess.run([
        "aider",
        "--model", f"ollama_chat/{model}",
        "--message", instruction,
        "--yes-always"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))


def main():
    print("--- Master Agent System Online ---")

    # Accept task from CLI args (automation) or fall back to interactive input
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        print(f"Task: {task}")
    else:
        task = input("Enter your system task: ")

    category = call_brain(task)
    target_model = AGENT_REGISTRY.get(category, AGENT_REGISTRY["coding"])

    print(f"\n> Brain Decision: {category.upper()} task detected.")
    print(f"> Routing to model: {target_model}")

    if category in ["coding", "security", "reasoning"]:
        launch_aider(target_model, task)
    else:
        subprocess.run(["ollama", "run", target_model, task])


if __name__ == "__main__":
    main()
