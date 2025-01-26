"""YouTube summarization graph that can be used as a tool by other graphs."""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.tools import YouTubeSearchTool
from shared.utils import load_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from retrieval_graph.configuration import AgentConfiguration

class YouTubeSummaryState(TypedDict):
    """State for the YouTube summarization workflow."""
    query: Optional[str]  # Search query if searching, None if direct URL
    video_url: Optional[str]  # Direct video URL if provided
    video_urls: List[str]  # Search results if searching
    current_video: str  # Current video being processed
    transcripts: List[Document]
    summary: str


# Add conditional edges based on routing
def route_to_next_node(state: YouTubeSummaryState):
    return state["next"]

def route_request(state: YouTubeSummaryState) -> Dict[str, str]:
    """Route the request based on whether we have a direct URL or need to search."""
    if state["video_url"]:
        return {"next": "extract_transcript"}  # Return a dict with the action
    return {"next": "search_videos"}  # Return a dict with the action


async def search_videos(state: YouTubeSummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Search for relevant YouTube videos if query provided."""
    youtube_search = YouTubeSearchTool()
    results = youtube_search.run(state["query"])
    
    # Convert string representation of list to actual list
    if isinstance(results, str):
        # Remove brackets and split by commas
        urls = results.strip('[]').replace("'", "").split(', ')
        results = [url.strip() for url in urls]
    
    return {
        "video_urls": results[:3],
        "current_video": results[0] if results else ""
    }


async def extract_transcript(state: YouTubeSummaryState, config: RunnableConfig) -> Dict[str, Any]:
    """Extract transcript from current video."""
    video_url = state["video_url"] or state["current_video"]
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )

    docs = loader.load_and_split(text_splitter)
    return {"transcripts": docs}


async def generate_summary(state: YouTubeSummaryState, config: RunnableConfig) -> Dict[str, str]:
    """Generate summary of the transcript."""

    configuration = AgentConfiguration.from_runnable_config(config)
    llm = load_chat_model(configuration.query_model)

    # Handle transcript whether it's a list or string
    transcripts = state["transcripts"]
    if isinstance(transcripts, list):
        transcripts = " ".join(doc.page_content for doc in transcripts)  # Join list elements into a single string

    # no need for LLM, waste of tokens
    # Embedeverything or summarize it?
    #       - waste of tokens?
    # Summarize it feature (for self use)
    prompt = f"""Summarize this YouTube video transcript concisely:
{transcripts}
Include: main points, key details, and takeaways."""
    
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt)

    cost =  (len(tokens) / 1000 * 0.005) + (len(tokens) / 1000 * 0.015)
    print(f"guessed price:{cost}")

    summary = await llm.ainvoke(prompt)
    return {"summary": summary.content}


def build_youtube_summary_graph() -> StateGraph:
    """Build the YouTube summary graph."""
    workflow = StateGraph(YouTubeSummaryState)
    
    # Add nodes
    workflow.add_node(route_request)
    workflow.add_node(search_videos)
    workflow.add_node(extract_transcript)
    workflow.add_node(generate_summary)

    # Add conditional routing
    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_to_next_node,
        # lambda state: state["next"],  # Use the 'next' field from state for routing
        {
           "extract_transcript": "extract_transcript",
           "search_videos": "search_videos"
        }
    )
    
    # Add remaining edges
    workflow.add_edge("search_videos", "generate_summary")
    workflow.add_edge("extract_transcript", "generate_summary")
    workflow.add_edge("generate_summary", END)  # Make sure all paths lead to END

    return workflow.compile()
