import requests

class Agent:
    def __init__(self, api_key: str, name:str = None):
        self.name = name
        self.api_key = api_key
        
    def query(self, prompt:str, model:str = "llama3.3:latest"):
        url = "https://ollama.nlpnchu.org/api/generate"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        while True:
            try: 
                r = requests.post(url, headers=headers, json=payload)
                # print(r.content) 
                r = r.json()
                return r["response"]
            except Exception as e: 
                print("Error:", e)