import os

from loader import UrlWalker, DocumentLoader

config = {
    "openai_api_key": os.environ['API_KEY'],
    "dbpath": "vectordb",
    "embedding_type": 'openai'
}

def test_load_single_url():
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

    loader = DocumentLoader(config)
    loader.get_db().get_vector_store().delete_collection()

    loader.load(
        urls
    )
    vector_store = loader.get_db().get_vector_store()
    assert len(vector_store) >= len(urls)

def test_web_crawler():
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    max_depth = 1

    crawler = UrlWalker(url, max_depth)
    crawler.crawl()
    urls = list(crawler.visited_urls)

    loader = DocumentLoader(config)
    loader.get_db().get_vector_store().delete_collection()

    loader.load(
        urls
    )
    vector_store = loader.get_db().get_vector_store()
    assert len(vector_store) >= len(urls)
