from copy import copy
from typing import Any, Optional, Sequence
from uuid import UUID

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document

from langchain import hub
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI


class RetrieverBase:
    """Base class for Retriever"""

    def __init__(self, dbpath):
        self.dbpath = dbpath

    def load_data(self):
        vectorstore = Chroma(persist_directory=self.dbpath,
                             embedding_function=self.get_embedding())
        retriever = vectorstore.as_retriever()
        return retriever

    def get_embedding(self):
        raise NotImplementedError("Derived classes must implement this method")


class OpenAIRetriever(RetrieverBase):
    """Retriever with OpenAI embeddings"""

    def __init__(self, openai_api_key, dbpath):
        super().__init__(dbpath)
        self.openai_api_key = openai_api_key

    def get_embedding(self):
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key)


class FastEmbedRetriever(RetrieverBase):
    """Retriever with FastEmbed embeddings"""

    def get_embedding(self):
        return FastEmbedEmbeddings()


class RetrieverFactory:
    """Factory class to create Retriever instances"""

    @staticmethod
    def create_retriever(config):
        embedding_type = config.get('embedding_type')
        dbpath = config.get('dbpath')
        if embedding_type == 'openai':
            openai_api_key = config.get('openai_api_key')
            return OpenAIRetriever(openai_api_key, dbpath)
        elif embedding_type == 'fastembed':
            return FastEmbedRetriever(dbpath)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")


def format_docs(docs, accumulator):
    accumulator.extend(docs)
    return "\n\n".join(doc.page_content for doc in docs)


class Response(object):
    def __init__(self, query, documents, output, history=None):
        self.query = query
        self.documents = documents
        self.output = output
        self.history = history


def create_llm(config):
    api_provider = config.get('api_provider')
    model = config.get('model')#"gpt-3.5-turbo"

    if api_provider == 'openai':
        return ChatOpenAI(model_name=model, temperature=0, openai_api_key=config.get("openai_api_key"))
    elif api_provider == 'ollama':
        return Ollama(model=model)
    else:
        raise ValueError(f"Unsupported embedding type: {api_provider}")


class Endpoint(object):
    def __init__(self, config):
        self._config = config
        self._retriever = RetrieverFactory.create_retriever(config)
        self._llm = create_llm(config)

    def get_response(self, query):
        prompt = hub.pull("rlm/rag-prompt")

        documents = []
        retriever = self._retriever.load_data()
        rag_chain = (
                {"context": retriever | (lambda x: format_docs(x, documents)), "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
        )
        output = rag_chain.invoke(query)

        return Response(query, documents, output)

    def get_response_with_docs(self, query, retriever):
        prompt = hub.pull("rlm/rag-prompt")

        documents = []
        rag_chain = (
                {"context": retriever | (lambda x: format_docs(x, documents)), "question": RunnablePassthrough()}
                | prompt
                | self._llm
                | StrOutputParser()
        )
        output = rag_chain.invoke(query)

        return Response(query, documents, output)

    def get_response_with_history(self, query, history):
        retriever = self._retriever.load_data()

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
        memory.chat_memory.add_messages(history)
        callback = CatchContextCallback()
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h
        )

        output = rag_chain.invoke(query, config=dict(callbacks=[callback]))

        return Response(query, callback.documents, output, memory.buffer_as_messages)


class CatchContextCallback(BaseCallbackHandler):
    def __init__(self):
        self.documents = None

    def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        self.documents = copy(documents)
