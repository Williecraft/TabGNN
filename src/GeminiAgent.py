from google import genai
from google.genai.errors import ClientError, ServerError
import time

class Agent:
    def __init__(self, api_key:list[str] | str, name:str = None):
        if type(api_key) == str: self.api_keys = [api_key]
        else: self.api_keys = api_key
        self.key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_index])
        self.name = name
        self.memory = ""

    def query(self, prompt:str, model:str = "gemini-2.5-flash", switch:bool=True):
        self.memory += prompt+"\n"+"-"*20+"\n"
        start_index = self.key_index
        while True:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    # contents=self.memory
                    contents=prompt
                )
                self.memory += f"{self.name}:\n"+response.text+"\n"+"-"*20+"\n"
                return response.text
            except ClientError:
                self.key_index = (self.key_index+1) % len(self.api_keys)
                if not switch:
                    print("Error: ClientError")
                    break
                if self.key_index == start_index:
                    print("Error: ClientError, all keys exhausted. Waiting 5 minutes...")
                    time.sleep(300)
                else:
                    print(f"Error: ClientError, switching to key index {self.key_index}.")
                    self.client = genai.Client(api_key=self.api_keys[self.key_index])
                    time.sleep(2)
            except ServerError:
                print("Error: ServerError, waiting 10 seconds...")
                time.sleep(10)