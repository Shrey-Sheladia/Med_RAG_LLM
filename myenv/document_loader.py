from pathlib import Path


class DocumentLoader:
    def __init__(self, path: Path | str, api_key: str):
        self._api_key = api_key
        self._doc_pages = []

        if isinstance(path, str):
            path = Path(path)

    def _load_documents_to_index(self):
        pass
