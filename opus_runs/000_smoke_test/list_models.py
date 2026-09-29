import os
import requests

key = os.environ["OPENROUTER_API_KEY"]
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=60,
)
response.raise_for_status()

for item in response.json()["data"]:
    model_id = item["id"].lower()
    if "opus" in model_id:
        print(item["id"])
