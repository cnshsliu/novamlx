"""Thinking/reasoning model example with NovaMLX-specific parameters."""

from novamlx import Client

client = Client()

response = client.chat.completions.create(
    model="Qwen3-8B-MLX-4bit",
    messages=[{"role": "user", "content": "What is 17 * 23? Think carefully."}],
    # NovaMLX-specific extensions
    thinking_budget=8192,
    enable_thinking=True,
    reasoning_effort="high",
)

msg = response.choices[0].message
if msg.reasoning_content:
    print(f"[Thinking]:\n{msg.reasoning_content}\n")
print(f"[Answer]: {msg.content}")
