import requests
LAB_KEY = "zhuantisheng"

url = "https://ollama.nlpnchu.org/api/generate"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LAB_KEY}"
}

payload = {
    "model": "llama3.3:latest",
    "prompt": "Hello",
    "stream": False
}

r = requests.post(url, headers=headers, json=payload)
r.raise_for_status()

data = r.json()
print(data)