from google import genai
from google.genai.errors import ClientError, ServerError
import time

class Agent:
    def __init__(self, api_keys:list[str] | str, name:str = None):
        if type(api_keys) == type(str): self.api_keys = [api_keys]
        else: self.api_keys = api_keys
        self.key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_index])
        self.name = name
        self.memory = ""

    def query(self, prompt:str, model:str = "gemini-2.5-flash"):
        self.memory += prompt+"\n"+"-"*20+"\n"
        while True:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=self.memory
                )
                self.memory += f"{self.name}:\n"+response.text+"\n"+"-"*20+"\n"
                return response.text
            except ClientError:
                self.key_index = (self.key_index+1) % len(self.api_keys)
                self.client = genai.Client(api_key=self.api_keys[self.key_index])
                time.sleep(2)
            except ServerError:
                time.sleep(10)