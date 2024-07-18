from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI

from prompt_format import SYS_PROMPT

load_dotenv()


class LanguageModel(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass

    @abstractmethod
    def reset(self, sys_prompt: str = None) -> None:
        pass


class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._messages = [
            {"role": "system", "content": SYS_PROMPT},
        ]

    def chat(self, prompt: str, stream: bool = False):
        self._messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._messages,
            stream=stream,
        )
        if not stream:
            response = response.choices[0].message.content
            self._messages.append({"role": "system", "content": response})

            return response
        else:
            full_message = ""
            for chunk in response:
                response_chunk = chunk.choices[0].delta.content
                full_message += response_chunk if response_chunk else ""
                yield response_chunk

            self._messages.append({"role": "system", "content": full_message})
            yield False

    def reset(self, sys_prompt: str = None) -> None:
        if not sys_prompt:
            sys_prompt = SYS_PROMPT
        self._messages = [
            {"role": "system", "content": sys_prompt},
        ]
