import streamlit as st
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# API Key Handling (use secrets if possible)
gemini_key = os.getenv('ASSEMBLYAI_API_KEY')

if gemini_key is None:
    gemini_key = st.text_input("Enter your Gemini API key:", type="password")

# Initialize OpenAI client and LLM
client = OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/")
llm = ChatOpenAI(
    model_name='gemini-1.5-flash',
    temperature=0.9,
    openai_api_key=gemini_key,
    openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Initialize session state variables
if 'paragraphs' not in st.session_state:
    st.session_state.paragraphs = []
if 'paragraph_embeddings' not in st.session_state:
    st.session_state.paragraph_embeddings = np.array([]) # Initialize as empty numpy array

# Function to embed text (no changes)
def embed_text(text):
    response = client.embeddings.create(input=text, model="text-embedding-004")
    return response.data[0].embedding

# Function to process text using session state
def process_text(text, chunk_size):
    st.session_state.paragraphs = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    with st.spinner("Calculating embeddings..."):
        st.session_state.paragraph_embeddings = [embed_text(paragraph) for paragraph in st.session_state.paragraphs]
    st.session_state.paragraph_embeddings = np.array(st.session_state.paragraph_embeddings)

# Streamlit App
st.title("Retrieval Augmented Generation (RAG) Demo")

uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

combined_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        combined_text += text

    st.text_area("Uploaded Text:", combined_text, height=150)

col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input("Chunk Size:", min_value=1, value=400)
with col2:
    if st.button("Process Text"):
        if not combined_text:
            st.info("Please upload text first.")
        else:
            process_text(combined_text, chunk_size)
            st.success("Text processed and embeddings calculated!")


query = st.text_input("Enter your query:")
n_results = st.number_input("Number of results to display:", min_value=1, value=1)
use_context = st.checkbox("Use Context", value=True)

col3, col4 = st.columns(2)
with col3:
    if st.button("Search"):
        if len(st.session_state.paragraph_embeddings) == 0:
            st.info("Please process the text first.")
        elif not query:
            st.info("Please enter a query.")
        else:
            query_embedding = embed_text(query)
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
            query_embedding = embed_text(query)
            similarities = cosine_similarity([query_embedding], st.session_state.paragraph_embeddings)
            most_similar_indices = np.argsort(similarities[0])[::-1][:n_results]

            if use_context:
                most_similar_paragraph = "\n\n".join([st.session_state.paragraphs[index] for index in most_similar_indices])
                llm_query = f"CONTEXT: {most_similar_paragraph}\n\nQUERY: {query}"
            else:
                llm_query = query

            messages = [
                {"role": "system", "content": "You are a reviewer who provides thorough answers to user queries."},
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