from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

def process_query(query, embeddings, paragraph_embeddings, query_embedding=None, n_results=1):
    if query_embedding is None:
        query_embedding = embeddings.embed_query(query)
    similarities = cosine_similarity([query_embedding], paragraph_embeddings)
    most_similar_indices = np.argsort(similarities[0])[::-1][:n_results]
    return most_similar_indices, query_embedding

def naive_rag_process(text, chunk_size, embeddings, query):
    # Get query embedding first
    query_embedding = embeddings.embed_query(query)
    
    # Process text
    paragraphs = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    with st.spinner("Calculating embeddings..."):
        paragraph_embeddings = embeddings.embed_documents(paragraphs)
    paragraph_embeddings = np.array(paragraph_embeddings)
    
    query_embedding = embeddings.embed_query(query)
    
    return paragraphs, paragraph_embeddings, query_embedding

def semantic_chunking_rag_process(text, chunk_size, embeddings, query):
    # For now, same as naive implementation
    return naive_rag_process(text, chunk_size, embeddings, query)

def reranker_rag_process(text, chunk_size, embeddings, query):
    # For now, same as naive implementation
    return naive_rag_process(text, chunk_size, embeddings, query)

def self_rag_process(text, chunk_size, embeddings, query):
    # For now, same as naive implementation
    return naive_rag_process(text, chunk_size, embeddings, query)

def graph_rag_process(text, chunk_size, embeddings, query):
    # For now, same as naive implementation
    return naive_rag_process(text, chunk_size, embeddings, query)

# Mapping of RAG types to their processing functions
RAG_IMPLEMENTATIONS = {
    "Naive RAG": naive_rag_process,
    "RAG with Semantic Chunking": semantic_chunking_rag_process,
    "RAG with Re-ranker": reranker_rag_process,
    "Self RAG": self_rag_process,
    "Graph RAG": graph_rag_process
}

# RAG descriptions for UI
RAG_DESCRIPTIONS = {
    "Naive RAG": "Basic RAG implementation with simple text chunking and embedding-based retrieval.",
    "RAG with Semantic Chunking": "RAG with intelligent text chunking based on semantic meaning.",
    "RAG with Re-ranker": "RAG with an additional re-ranking step to improve retrieval relevance.",
    "Self RAG": "Self-reflective RAG that can generate and verify its own context.",
    "Graph RAG": "Graph-based RAG that maintains relationships between chunks of text."
}