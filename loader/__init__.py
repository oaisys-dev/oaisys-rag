import copy
from collections import deque
from typing import List

import bs4
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Class that runs crawling jobs. Job config is provided to the run method.
    Crawler uses appropriate document loader to parse documents. Then splits them into chunks and
    """

    def __init__(self, config):
        self.db = DocumentStoreFactory.create_document_store(config)

    def load(self, urls):
        loader = self._loader(urls)

        docs = loader.load()
        self.db.save(docs)

    def _loader(self, urls):
        return WebBaseLoader(
            web_paths=urls,
            # bs_kwargs=dict(
            #     parse_only=bs4.SoupStrainer(
            #         class_=("post-content", "post-title", "post-header")
            #     )
            # ),
        )

    def get_passthrough_retriver(self, urls):
        loader = self._loader(urls)

        retriever = PassThroughRetriever(documents = loader.load())

        return retriever

    def get_db(self):
        return self.db


class DocumentStoreBase:
    """Base class for DocumentStore"""

    def __init__(self, dbpath):
        self.dbpath = dbpath

    def save(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        Chroma.from_documents(persist_directory=self.dbpath,
                              documents=splits, embedding=self.get_embedding())

    def clear(self):
        vectorstore = Chroma(persist_directory=self.dbpath,
                             embedding_function=self.get_embedding())
        vectorstore.delete_collection()

    def get_vector_store(self):
        vectorstore = Chroma(persist_directory=self.dbpath,
                             embedding_function=self.get_embedding())
        return vectorstore

    def get_embedding(self):
        raise NotImplementedError("Derived classes must implement this method")


class OpenAIDocumentStore(DocumentStoreBase):
    """DocumentStore with OpenAI embeddings"""

    def __init__(self, openai_api_key, dbpath):
        super().__init__(dbpath)
        self.openai_api_key = openai_api_key

    def get_embedding(self):
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key)


class FastEmbedDocumentStore(DocumentStoreBase):
    """DocumentStore with FastEmbed embeddings"""

    def get_embedding(self):
        return FastEmbedEmbeddings()


class DocumentStoreFactory:
    """Factory class to create DocumentStore instances"""

    @staticmethod
    def create_document_store(config):
        embedding_type = config.get('embedding_type')
        dbpath = config.get('dbpath')
        if embedding_type == 'openai':
            openai_api_key = config.get('openai_api_key')
            return OpenAIDocumentStore(openai_api_key, dbpath)
        elif embedding_type == 'fastembed':
            return FastEmbedDocumentStore(dbpath)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")


class UrlWalker:
    def __init__(self, base_url, max_depth):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited_urls = set()
        self.local_domain = urlparse(base_url).netloc

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def crawl(self):
        queue = deque([(self.base_url, 0)])

        while queue:
            current_url, depth = queue.popleft()
            if depth > self.max_depth or current_url in self.visited_urls:
                continue

            try:
                response = requests.get(current_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch {current_url}: {e}")
                continue

            self.visited_urls.add(current_url)
            self.logger.debug(f"Depth: {depth}, URL: {current_url}")

            soup = BeautifulSoup(response.content, 'html.parser')

            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if self._is_valid_url(href):
                    full_url = urljoin(current_url, href)
                    if self._is_local_url(full_url):
                        queue.append((full_url, depth + 1))

    def _is_valid_url(self, url):
        # Check if the URL is valid and not a fragment or mailto link
        return url and not url.startswith('#') and not url.startswith('mailto:')

    def _is_local_url(self, url):
        # Check if the URL is local to the base URL's domain
        return urlparse(url).netloc == self.local_domain


class PassThroughRetriever(BaseRetriever):
    """A retriever that returns all documents no matter what is the user query.

    This retriever only implements the sync method _get_relevant_documents.
    """
    documents: List[Document]

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        return copy.copy(self.documents)