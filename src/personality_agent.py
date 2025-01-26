from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from index_graph import graph as index_graph
from index_graph.state import IndexState

class PersonalityCloneState(TypedDict):
    # Enhanced state to capture more context
    input: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    personality_profile: Dict[str, Any]
    retrieved_context: List[Document]
    research_plan: Optional[List[str]]
    current_research_step: Optional[int]
    interaction_summary: List[Dict]
    emotional_state: Dict[str, float]

class PersonalityCloneGraph:
    def __init__(self, documents_path, model="gpt-4-turbo"):
        # Initialize core components
        self.documents = self.load_personality_documents(documents_path)
        self.vectorstore = self.create_vector_store(self.documents)
        self.personality_profile = self.generate_personality_profile()
        self.llm = ChatOpenAI(model=model)
        
        # Build the graph
        self.graph = self.build_personality_clone_graph()

    def load_personality_documents(self, directory_path):
        """Enhanced document loading with metadata extraction"""
        loader = DirectoryLoader(directory_path, recursive=True)
        documents = loader.load()
        
        index_graph.index_docs(state=IndexState(docs=documents))
        from shared.retrieval import make_retriever
        
        make_retriever()


        return split_docs

    def generate_personality_profile(self):
        """Advanced personality profile generation"""
        profile_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze these documents comprehensively to create a multi-dimensional personality profile.
            Include the following dimensions:
            1. Communication Style
            2. Core Values
            3. Knowledge Domains
            4. Emotional Intelligence
            5. Unique Perspectives
            6. Potential Biases
            7. Learning Approach
            
            Provide a structured JSON output."""),
            ("human", "{documents}")
        ])
        
        chain = profile_prompt | self.llm | StrOutputParser()
        profile = chain.invoke({"documents": self.documents})
        return json.loads(profile)

    def create_vector_store(self, documents):
        """Enhanced vector store with metadata filtering"""
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore

    def build_personality_clone_graph(self):
        graph = StateGraph(PersonalityCloneState)
        
        # Nodes Definition
        def analyze_query(state):
            """Analyze and route the query based on complexity"""
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the query and determine:
                1. Query Type (General/Complex/Research)
                2. Required Depth of Response
                3. Potential Research Steps
                Output a structured analysis."""),
                ("human", "{input}")
            ])
            
            analysis_chain = analysis_prompt | self.llm | StrOutputParser()
            query_analysis = analysis_chain.invoke({"input": state['input']})
            
            return {
                "query_type": query_analysis.get('type', 'general'),
                "research_plan": query_analysis.get('research_steps', [])
            }
        
        def retrieve_context(state):
            """Advanced context retrieval with semantic and metadata-based search"""
            query = state['input']
            retrieved_context = self.vector
