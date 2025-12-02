import json
from GeminiAgent import Agent

with open("/user_data/TabGNN/config/api_keys.json", "r", encoding="utf-8") as f:
    api_keys = json.load(f)

for i, key in enumerate(api_keys):
    agent = Agent(api_keys=key)
    print(f"Testing key {i}: {key}")
    print(agent.query("Hello", switch=False))