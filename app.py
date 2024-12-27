# https://python.langchain.com/docs/versions/v0_3/
# https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI

import streamlit as st
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

# API Key Handling (use secrets if possible)
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

# Dev mode
# GOOGLE_API_KEY = 

if GOOGLE_API_KEY is None:
    GOOGLE_API_KEY = st.text_input("Enter your Gemini API key:", type="password")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Initialize OpenAI client and LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize session state variables
if 'paragraphs' not in st.session_state:
    st.session_state.paragraphs = []
if 'paragraph_embeddings' not in st.session_state:
    st.session_state.paragraph_embeddings = np.array([])

# RAG Implementation Functions
def naive_rag_process(text, chunk_size):
    st.session_state.paragraphs = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    with st.spinner("Calculating embeddings..."):
        st.session_state.paragraph_embeddings = embeddings.embed_documents(st.session_state.paragraphs)
    st.session_state.paragraph_embeddings = np.array(st.session_state.paragraph_embeddings)

def semantic_chunking_rag_process(text, chunk_size):
    # For now, same as naive implementation
    naive_rag_process(text, chunk_size)

def reranker_rag_process(text, chunk_size):
    # For now, same as naive implementation
    naive_rag_process(text, chunk_size)

def self_rag_process(text, chunk_size):
    # For now, same as naive implementation
    naive_rag_process(text, chunk_size)

def graph_rag_process(text, chunk_size):
    # For now, same as naive implementation
    naive_rag_process(text, chunk_size)

# Mapping of RAG types to their processing functions
RAG_IMPLEMENTATIONS = {
    "Naive RAG": naive_rag_process,
    "RAG with Semantic Chunking": semantic_chunking_rag_process,
    "RAG with Re-ranker": reranker_rag_process,
    "Self RAG": self_rag_process,
    "Graph RAG": graph_rag_process
}

# Streamlit App
st.title("Retrieval Augmented Generation (RAG) Demo")

# Add RAG type selector
rag_type = st.selectbox(
    "Select RAG Implementation:",
    list(RAG_IMPLEMENTATIONS.keys()),
    help="Choose the type of RAG implementation to use"
)

# Display information about the selected RAG type
rag_descriptions = {
    "Naive RAG": "Basic RAG implementation with simple text chunking and embedding-based retrieval.",
    "RAG with Semantic Chunking": "RAG with intelligent text chunking based on semantic meaning.",
    "RAG with Re-ranker": "RAG with an additional re-ranking step to improve retrieval relevance.",
    "Self RAG": "Self-reflective RAG that can generate and verify its own context.",
    "Graph RAG": "Graph-based RAG that maintains relationships between chunks of text."
}
st.info(rag_descriptions[rag_type])

uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

st.session_state.combined_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        st.session_state.combined_text += text

    st.text_area("Uploaded Text:", st.session_state.combined_text, height=150)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input("Chunk Size:", min_value=1, value=128)
with col2:
    if st.button("Process Text"):
        if not st.session_state.combined_text:
            st.info("Please upload text first.")
        else:
            # Use the selected RAG implementation
            process_function = RAG_IMPLEMENTATIONS[rag_type]
            process_function(st.session_state.combined_text, chunk_size)
            st.success(f"Text processed using {rag_type}!")

query = st.text_input("Enter your query:")
n_results = st.number_input("Number of results to display:", min_value=1, value=1)
use_context = st.checkbox("Use Context", value=True)

system_message = {
                "role": "system",
                "content": """You are an AI assistant specifically designed to work with RAG (Retrieval Augmented Generation) systems. Your responses must be strictly based on the provided context. Follow these rules:

            1. Only answer using information explicitly present in the given context
            2. If the context is not relevant to the query, respond with "The provided context is not relevant to answer this query."
            3. If the context is partially relevant but incomplete, respond with "The context is incomplete to fully answer this query. Based on the available context, I can only tell you that: [insert relevant information from context]"
            4. Do not make assumptions or add information beyond what's in the context
            5. If you quote or reference specific parts of the context, indicate this in your response
            6. If the context is empty or missing, respond with "No context was provided to answer this query."

            Remember: You are a context-dependent system. Never draw from general knowledge - only use the specific context provided in each query."""
            }

col3, col4 = st.columns(2)
with col3:
    if st.button("Search"):
        if len(st.session_state.paragraph_embeddings) == 0:
            st.info("Please process the text first.")
        elif not query:
            st.info("Please enter a query.")
        else:
            query_embedding = embeddings.embed_query(query)
            similarities = cosine_similarity([query_embedding], st.session_state.paragraph_embeddings)
            most_similar_indices = np.argsort(similarities[0])[::-1][:n_results]
            st.subheader("Search Results:")
            for index in most_similar_indices:
                st.write(st.session_state.paragraphs[index])

with col4:
    if st.button("Run Query"):
        if not st.session_state.paragraph_embeddings.size:
            st.info("Please process the text first.")
        elif not query:
            st.info("Please enter a query.")
        else:
            query_embedding = embeddings.embed_query(query)
            similarities = cosine_similarity([query_embedding], st.session_state.paragraph_embeddings)
            most_similar_indices = np.argsort(similarities[0])[::-1][:n_results]

            if use_context:
                most_similar_paragraph = "\n\n".join([st.session_state.paragraphs[index] for index in most_similar_indices])
                llm_query = f"CONTEXT: {most_similar_paragraph}\n\nQUERY: {query}"
            else:
                llm_query = query

            messages = [system_message,
                {"role": "user", "content": llm_query},
            ]

            try:
                response = llm.invoke(messages).content
                st.write(response)

                if use_context:
                    st.subheader("Relevant Context:")
                    for index in most_similar_indices:
                        st.write(st.session_state.paragraphs[index])

            except Exception as e:
                st.error(f"Error generating response: {e}")