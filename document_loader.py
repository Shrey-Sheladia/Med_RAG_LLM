import logging
from pathlib import Path

from autologging import logged
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


@logged
class DocumentStore:
    def __init__(self, path: Path | str, api_key: str):
        self._api_key = api_key
        self._doc_pages = []
        self.index = None
        self._embeddings = OpenAIEmbeddings(api_key=self._api_key)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        if isinstance(path, str):
            path = Path(path)
        self._path = path

        self._load_docs()

    def load_documents_to_index(self):
        self.index = FAISS.from_documents(
            documents=self._doc_pages, embedding=self._embeddings
        )
        logging.critical(f"Indexed {len(self._doc_pages)} documents")

    def search(self, query: str, top_k: int = 5):
        return self.index.similarity_search(query, top_k)

    def _load_docs(self):
        for file in self._path.iterdir():
            if file.suffix == ".pdf":
                page_content_docs = self._get_text_from_pdf(file)
                logging.critical(
                    f"Loaded {len(page_content_docs)} pages from {file.name}"
                )
            else:
                raise NotImplementedError("Only PDFs are supported")

            self._doc_pages.extend(page_content_docs)

    def _get_text_from_pdf(self, file: Path):
        loader = PyPDFLoader(file.as_posix())
        chunks = loader.load_and_split(text_splitter=self._splitter)
        return chunks


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    logging.basicConfig(level=logging.CRITICAL)

    load_dotenv()

    folder = Path("./docs")
    API_KEY = os.environ.get("OPEN_AI_KEY")

    docs_store = DocumentStore(path=folder, api_key=API_KEY)

    docs_store.load_documents_to_index()
    query = "What is wizards chess?"
    results = docs_store.search(query)
    for result in results:
        print(result)
        print("\n\n\n")
