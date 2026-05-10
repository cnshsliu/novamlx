"""Tool calling example."""

from novamlx import Client

client = Client()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="Qwen3-8B-MLX-4bit",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)

choice = response.choices[0]
if choice.message.tool_calls:
    for tc in choice.message.tool_calls:
        print(f"Tool: {tc.function.name}")
        print(f"Args: {tc.function.arguments}")
