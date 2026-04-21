from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

"""Module for managing a vector store using FAISS."""


class VectorStore:
    """
    A class to manage a vector database using FAISS. 
    It allows adding documents, building an index, and performing similarity searches. 
    The documents are stored as pickled files in a specified directory, and the 
    FAISS index is saved to a specified path.

    Attributes:
        index_path (Path): The directory where the FAISS index will be saved and loaded from.
        doc_path (Path): The directory where the document pickled files will be stored and loaded from.
        dimension (int): The dimensionality of the vectors in the FAISS index.
        # TODO: consider sharding the index and documents for scalability
    """
    def __init__(
        self,
        index_path: Path,
        doc_path: Path,
        dimension: int = 128,
        *,
        on_after_add: Callable[[], None] | None = None,
    ) -> None:
        self._on_after_add = on_after_add
        self.doc_path = doc_path
        self.documents = {}
        # Create the document directory if it doesn't exist, otherwise load existing docstore
        if not self.doc_path.exists():
            self.doc_path.mkdir(parents=True, exist_ok=True)
        else:
            docstore_file = self.doc_path / "docstore.pkl"
            if docstore_file.exists():
                with open(docstore_file, "rb") as f:
                    self.documents = pickle.load(f)

        self.index = None
        self.index_path = index_path
        self.dimension = dimension
        # Create the index directory if it doesn't exist, otherwise load existing index
        if not self.index_path.exists():
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            index_file = self.index_path / "index.bin"
            if index_file.exists():
                self.index = faiss.read_index(str(self.index_path / "index.bin"))
            else:
                self.index = faiss.IndexFlatL2(self.dimension)


    def add(self, documents: list[str], embeddings: np.ndarray, metadata: list[dict]=None) -> None:

        """
        Add text and its corresponding metadata to the document store.
        Add the corresponding vector to the FAISS index.
        Args:
            documents (list[str]): A list of text strings to be added.
            embeddings (np.ndarray): A numpy array of vector embeddings corresponding to the documents being added.
            metadata (list[dict], optional): A list of metadata dictionaries corresponding to the documents being added.
                Use None entries for documents with no metadata. Defaults to None (all empty).
        """

        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")

        if metadata and len(metadata) != len(documents):
            raise ValueError("The number of metadata entries must match the number of documents.")

        if metadata is None:
            metadata = [None] * len(documents)

        try:
            start_id = self.index.ntotal
            for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, metadata)):
                faiss_id = start_id + i
                self.documents[faiss_id] = {"text": doc, "metadata": meta}
                self.index.add(embedding.reshape(1, -1))
            if self._on_after_add is not None:
                self._on_after_add()
        except Exception as e:
            raise RuntimeError(f"Failed to add documents and/or embeddings to the index: {e}")

    def save(self) -> None:
        """
        Save the FAISS index and the document store to disk. 
        The FAISS index is saved as a binary file, and the documents are saved as pickled files in the specified directory.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized. Cannot save an uninitialized index.")

        try:
            faiss.write_index(self.index, str(self.index_path / "index.bin"))
            with open(self.doc_path / "docstore.pkl", "wb") as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save the index and documents: {e}")

    def load(self) -> None:
        """
        Load the FAISS index and the document store from disk.
        The FAISS index is loaded from a binary file, and the documents are loaded from pickled files in the specified directory.
        """
        try:
            self.index = faiss.read_index(str(self.index_path / "index.bin"))
            with open(self.doc_path / "docstore.pkl", "rb") as f:
                self.documents = pickle.load(f)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load the index and documents: {e}")

    def search(self, query_embedding: list[float], top_k: int=5) -> list[dict]:
        """
        Perform a similarity search in the FAISS index using the provided query embedding. 
        Returns the top K most similar documents along with their metadata.
        Args:
            query_embedding (list[float]): The vector embedding of the query text.
            top_k (int, optional): The number of top similar documents to return. Defaults to 5.
        Returns:
            list[dict]: A list of dictionaries containing the text and metadata of the top K similar documents.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized. Cannot perform search on an uninitialized index.")

        try:
            D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
            results = []
            for faiss_id, distance in zip(I[0], D[0]):
                if faiss_id in self.documents:
                    entry = self.documents[faiss_id]
                    results.append({**entry, "distance": float(distance)})
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to perform search: {e}")
