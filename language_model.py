from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LanguageModel(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass


class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def complete(self, prompt: str) -> str:
        self._messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._messages,
            stream=False,
        )
        response = response.choices[0].message.content
        self._messages.append({"role": "system", "content": response})

        return response

    def reset(self, prompt: str = None) -> None:
        if not prompt:
            prompt = "You are a helpful assistant."
        self._messages = [
            {"role": "system", "content": prompt},
        ]
