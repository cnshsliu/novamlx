"""Streaming chat completion example."""

from novamlx import Client

client = Client()

for chunk in client.chat.completions.create(
    model="Qwen3-8B-MLX-4bit",
    messages=[{"role": "user", "content": "Write a haiku about programming."}],
    stream=True,
):
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    if delta.reasoning_content:
        print(f"\n[thinking] {delta.reasoning_content}", end="", flush=True)

print()
