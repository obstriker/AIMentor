import os
from contextlib import contextmanager
from typing import Generator
from langchain_community.callbacks.manager import get_openai_callback

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langsmith import expect, unit

from index_graph import graph as index_graph
from retrieval_graph import graph
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder
from src.youtube_summarize import build_youtube_summary_graph, YouTubeSummaryState
from src.retrieval_graph.configuration import AgentConfiguration
from dotenv import load_dotenv

@contextmanager
def make_elastic_vectorstore(
    configuration: BaseConfiguration,
) -> Generator[VectorStore, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    embedding_model = make_text_encoder(configuration.embedding_model)
    vstore = ElasticsearchStore(
        es_user=os.environ["ELASTICSEARCH_USER"],
        es_password=os.environ["ELASTICSEARCH_PASSWORD"],
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )
    yield vstore


@pytest.mark.asyncio
@unit
async def test_retrieval_graph() -> None:
    simple_doc = 'In LangGraph, nodes are typically python functions (sync or async) where the first positional argument is the state, and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).'
    config = RunnableConfig(
        configurable={
            "retriever_provider": "elastic-local",
            "embedding_model": "openai/text-embedding-3-small",
        }
    )
    configuration = BaseConfiguration.from_runnable_config(config)

    doc_id = "test_id"
    result = await index_graph.ainvoke(
        {"docs": [{"page_content": simple_doc, "id": doc_id}]}, config
    )
    expect(result["docs"]).against(lambda x: not x)  # we delete after the end
    # test general query
    res = await graph.ainvoke(
        {"messages": [("user", "Hi! How are you?")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("general")

    # test query that needs more info
    res = await graph.ainvoke(
        {"messages": [("user", "I am having issues with the tools")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("more-info")

    # test LangChain-related query
    res = await graph.ainvoke(
        {"messages": [("user", "What is a node in LangGraph?")]},
        config,
    )
    expect(res["router"]["type"]).to_contain("langchain")
    response = str(res["messages"][-1].content)
    expect(response.lower()).to_contain("function")

    # clean up after test
    with make_elastic_vectorstore(configuration) as vstore:
        await vstore.adelete([doc_id])


@pytest.mark.asyncio
async def test_youtube_summary_graph_search():
    """Test YouTube summary graph with search path."""
    graph = build_youtube_summary_graph()
    graph.name = "YouTubeSummaryGraph"
    
    config = RunnableConfig(
        configurable={
            "model": "gpt-3.5-turbo",
        }
    )
    
    initial_state: YouTubeSummaryState = {
        "query": "langchain tutorial short",
        "video_url": None,
        "video_urls": [],
        "current_video": "",
        "transcript": "",
        "summary": ""
    }
    
    result = await graph.ainvoke(initial_state, config)
    print(result)

    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0
    assert len(result["video_urls"]) > 0


@pytest.mark.asyncio
async def test_youtube_summary_graph_direct_url():
    """Test YouTube summary graph with direct URL path."""
    graph = build_youtube_summary_graph()
    graph.name = "YouTubeSummaryGraph"
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    config = RunnableConfig(
        configurable={
            "query_model": "openai/gpt-4o-mini",
        }
    )

    initial_state: YouTubeSummaryState = {
        "query": None,
        "video_url": "https://www.youtube.com/watch?v=aywZrzNaKjs",
        "video_urls": [],
        "current_video": "",
        "transcript": "",
        "summary": ""
    }
    
    result = await graph.ainvoke(initial_state, config)
    print(result)

    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0
    assert not result["video_urls"]  # Should be empty since we used direct URL
