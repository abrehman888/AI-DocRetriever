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

# Set page configuration for title and icon
st.set_page_config(page_title="Chat with Xeven Solution ", page_icon=":speech_balloon:")

# Load environment variables
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_key = st.secrets["QDRANT_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
collection_name = st.secrets["Collection_Name"]
llm_name = "gpt-4o-mini"

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Create a QdrantClient instance
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Initialize QdrantVectorStore
qdrant = QdrantVectorStore(client=client, embedding=embed_model, collection_name=collection_name)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        body {
            font-family: 'Open Sans', sans-serif;
            background-color: #f7f9fc;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .logo-img {
            width: 50px;
        }

        .input-area {
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #fff;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stTextInput, .stButton button {
            font-size: 18px;
        }
        .stTextInput {
            padding: 10px;
            border-radius: 5px;
        }

        .response-bubble {
            margin-top: 10px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 10px;
            color: #333;
        }

        .stButton button {
            background-color: #4285f4;
            color: white;
            border-radius: 5px;
            padding: 10px 15px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #1a73e8;
        }

        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 14px;
            color: #888;
        }

        /* Highlight the name */
        .highlighted-name {
            font-weight: bold;
            color: #007bff;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="header-title"><img src="https://raw.githubusercontent.com/abrehman888/RAG/refs/heads/main/xevensolutions_logo.jpeg" class="logo-img" />Chat with Xeven Solution</div>', unsafe_allow_html=True)
st.write("**Developed by <span class='highlighted-name'>Abdul Rehman</span>**")

st.write("Ask your question below:")

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
chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)
query_fetcher = itemgetter("question")
setup = {"question": query_fetcher, "context": query_fetcher | retriever | format_docs}
_chain = setup | _prompt | chat_llm | StrOutputParser()

# Create a form for user input
with st.form(key="query_form"):
    query = st.text_input("Ask a question about Xeven:", key="user_query")
    submit_button = st.form_submit_button(label=" 💡 Get Response")  # Added submit button

# Process the query if the form is submitted
if submit_button and query:
    response = _chain.invoke({"question": query})
    st.markdown(f'<div class="response-bubble">Response: {response}</div>', unsafe_allow_html=True)

# Option to clear the chat history
if st.button("🧹 Clear History"):
    st.session_state['chat_history'] = []
    st.success("Chat history cleared!")

st.markdown("---")  # Adds a line separator
st.markdown('<div class="footer">Developed by <span class="highlighted-name">Abdul Rehman</span>. Powered by Xeven Solutions.</div>', unsafe_allow_html=True)
