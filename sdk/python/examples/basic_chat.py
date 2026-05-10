"""Basic chat completion example."""

import novamlx

# Zero-config: auto-discovers from ~/.nova/config.json or env vars
response = novamlx.chat.completions.create(
    model="Qwen3-8B-MLX-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one paragraph."},
    ],
    temperature=0.7,
    max_tokens=512,
)

print(response.choices[0].message.content)
print(f"\nUsage: {response.usage}")
