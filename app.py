import streamlit as st
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import openai
from langchain.chains import create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Load environment variables
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_API_KEY"]
collection_name = st.secrets["Collection_Name"]
llm_name = "gpt-4o-mini"
openai_api_key=st.secrets["OPENAI_API_KEY"]

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create a QdrantClient instance
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Initialize QdrantVectorStore
qdrant = QdrantVectorStore(client=client, embedding=embed_model, collection_name='demo')

# Streamlit UI
st.image("https://raw.githubusercontent.com/abrehman888/RAG/refs/heads/main/xevensolutions_logo.jpeg", width=100)
st.markdown("<h1 style='text-align: center; font-weight: bold;'> 💬 Chat with Xeven Solution</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: grey;'>Developed by <span style='color: #D83A3A;'>Abdul Rehman</span></p>", unsafe_allow_html=True)

# User query input
query = st.text_input("🔍 Ask a question about Xeven:")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the prompt template
prompt_str = """
Answer the user question based only on the following context:
{context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)
num_chunks = 3
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

# Check if the user has entered their API key

chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai_api_key, temperature=0)
query_fetcher = itemgetter("question")
setup = {"question": query_fetcher, "context": query_fetcher | retriever | format_docs}
_chain = setup | _prompt | chat_llm | StrOutputParser()

    # Get Response button
    if st.button("💡 Get Response"):
        if query:
            response = _chain.invoke({"question": query})
            st.write("Response:", response)
        else:
            st.warning("Please enter a question to get a response.")

# Option to clear the chat history
if st.button("🧹 Clear History", key="clear_history", help="Clear the chat history to start fresh"):
    st.session_state['chat_history'] = []
    st.success("Chat history cleared!")

# Additional footer styling and separator
st.markdown("<hr style='border: 1px solid #DDD;'/>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'> 🚀 Developed by Abdul Rehman. Powered by Xeven Solutions.</p>", unsafe_allow_html=True)
