import streamlit as st
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai
import os
from langchain.chains import create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
collection_name=st.secrets["Collection_Name"]
# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create a QdrantClient instance
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Initialize QdrantVectorStore
qdrant = QdrantVectorStore(client=client, embedding=embed_model, collection_name='demo')

# Streamlit UI
st.title("Chat with Qdrant and OpenAI")
st.write("Ask your question below:")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("Your Question:")

if st.button("Submit"):
    if query:
        # Store the user query in chat history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Create a ChatOpenAI instance
        chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai.api_key)

        # Get the response from the model using the chat history
        response = chat_llm(st.session_state.chat_history)

        # Store the AI response in chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

        # Display the response
        st.write("Response:")
        st.write(response['choices'][0]['message']['content'])

        # Display chat history
        st.write("Chat History:")
        for msg in st.session_state.chat_history:
            st.write(f"{msg['role']}: {msg['content']}")
    else:
        st.error("Please enter a question.")
