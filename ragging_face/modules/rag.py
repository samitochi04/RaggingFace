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
from utils.file_utils import extract_text_from_file

def ingest_files(file_paths: List[str], rag_store: RAGStore):
    texts = []
    metas = []
    for path in file_paths:
        text = extract_text_from_file(path)
        if text:
            texts.append(text)
            metas.append({"source": os.path.basename(path)})
    if texts:
        rag_store.from_texts(texts, metadatas=metas)
