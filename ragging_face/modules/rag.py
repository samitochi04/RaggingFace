import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


class RAGStore:
    def __init__(self, persist_path: str = None):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index = None
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            try:
                self.index = FAISS.load_local(persist_path, self.embeddings)
            except Exception:
                self.index = None

    def add_documents(self, docs: List[Document]):
        """Add a list of langchain Document objects to the vector store."""
        if self.index is None:
            self.index = FAISS.from_documents(docs, self.embeddings)
        else:
            self.index.add_documents(docs)
        if self.persist_path:
            self.index.save_local(self.persist_path)

    def from_texts(self, texts: List[str], metadatas: List[dict] = None):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents(texts, metadatas=metadatas)
        self.add_documents(docs)

    def query(self, query_text: str, k: int = 4):
        if self.index is None:
            return []
        return self.index.similarity_search(query_text, k=k)


# helper to read pdf or text
# perform import lazily using relative package path

def ingest_files(file_paths: List[str], rag_store: RAGStore):
    # dynamically load the utility module to avoid import path issues
    import importlib.util
    util_path = os.path.join(os.path.dirname(__file__), '../utils/file_utils.py')
    util_path = os.path.abspath(util_path)
    spec = importlib.util.spec_from_file_location('file_utils', util_path)
    file_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_utils)  # type: ignore
    extract_text_from_file = file_utils.extract_text_from_file

    texts = []
    metas = []
    for path in file_paths:
        text = extract_text_from_file(path)
        if text:
            texts.append(text)
            metas.append({"source": os.path.basename(path)})
    if texts:
        rag_store.from_texts(texts, metadatas=metas)
