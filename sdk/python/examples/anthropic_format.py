"""Anthropic-format Messages API example."""

from novamlx import Client

client = Client()

# Non-streaming Anthropic format
response = client.messages.create(
    model="Qwen3-8B-MLX-4bit",
    max_tokens=1024,
    system="You are a concise assistant.",
    messages=[{"role": "user", "content": "Explain neural networks in 2 sentences."}],
    # NovaMLX extensions work in Anthropic format too
    thinking_budget=4096,
    enable_thinking=True,
)

for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking]: {block.thinking}")
    elif block.type == "text":
        print(f"[Answer]: {block.text}")

print(f"\nUsage: {response.usage}")
print(f"Stop reason: {response.stop_reason}")
