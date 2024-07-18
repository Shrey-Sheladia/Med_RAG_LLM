import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from document_loader import DocumentStore
from language_model import LanguageModel, OpenAILanguageModel
from prompt_format import PROMPT_FORMAT


class RAGLanguageModel:
    def __init__(
        self,
        api_key: str,
        document_folder: Path | str,
        llm: LanguageModel,
        document_store: DocumentStore,
    ):
        self._api_key = api_key
        self._path = document_folder
        self._llm = llm(api_key=self._api_key)
        self._document_store = document_store(path=self._path, api_key=self._api_key)
        self._document_store.load_documents_to_index()

    def start_conversation(self) -> str:
        while True:
            question = input("Input: ")
            if question.lower() == "q":
                break

            context = self._document_store.search(question)
            formatted_context = self._format_context(context)
            prompt = PROMPT_FORMAT.format(formatted_context, question)
            response = self._llm.chat(prompt=prompt, stream=True)
            for chunk in response:
                if chunk:
                    print(chunk, end="")
                elif chunk is False:
                    print("\n")
                    break

    def _format_context(self, context: list) -> str:
        formatted_contect = ""
        for index, document in enumerate(context):
            formatted_contect += f"Document {index + 1})\n"
            formatted_contect += document.page_content
            formatted_contect += "\n\n"

        return formatted_contect


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    load_dotenv()

    folder = Path("./docs")
    API_KEY = os.environ.get("OPEN_AI_KEY")

    rag = RAGLanguageModel(
        api_key=API_KEY,
        document_folder=folder,
        llm=OpenAILanguageModel,
        document_store=DocumentStore,
    )
    rag.start_conversation()
