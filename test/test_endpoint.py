import os

from endpoint import Endpoint
from loader import DocumentLoader

config = {
    "openai_api_key": os.environ['API_KEY'],
    "dbpath": "vectordb",
    "embedding_type": 'openai',
    "api_provider": "openai",
    "model": "gpt-4o-mini"
}

config_ollama = {
    "openai_api_key": os.environ['API_KEY'],
    "dbpath": "vectordb",
    "embedding_type": 'openai',
    "api_provider": "ollama",
    "model": "phi3"
}

def test_endpoint():
    ep = Endpoint(config)
    query = "How LLM agent memory work?"
    result = ep.get_response(query)
    assert result.query == query
    assert len(result.documents) > 0
    assert result.output

def test_endpoint_with_docs():
    urls = ["https://meduza.io/rss/en/all"]

    loader = DocumentLoader(config)

    passthrough_retriver = loader.get_passthrough_retriver(
        urls
    )

    ep = Endpoint(config)
    query = "What is the latest in Ukraine?"
    result = ep.get_response_with_docs(query, passthrough_retriver)
    assert result.query == query
    assert len(result.documents) > 0
    assert result.output

def test_endpoint_with_history():
    ep = Endpoint(config)

    result = ep.get_response_with_history("How LLM agent memory work?", [])

    query = 'How it can be implemented?'
    result = ep.get_response_with_history(query, result.history)
    assert result.query == query
    assert len(result.documents) > 0
    assert result.output

def test_endpoint_ollama():
    ep = Endpoint(config_ollama)
    query = "How LLM agent memory work?"
    result = ep.get_response(query)
    assert result.query == query
    assert len(result.documents) > 0
    assert result.output
