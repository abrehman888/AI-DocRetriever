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
from langchain.schema import AIMessage, HumanMessage

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

# Initialize session state for history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the prompt template
prompt_str = """
Answer the user question based only on the following context:

conversation_history: {chat_history}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Number of chunks for retrieval
num_chunks = 3

# Qdrant retriever setup (adjust 'qdrant' to your Qdrant client configuration)
retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

# LLM setup
llm_name = "gpt-3.5-turbo"
chat_llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)

# Extractors for question and chat history
query_fetcher = itemgetter("question")
history_fetcher = itemgetter("chat_history")

# Streamlit app layout
st.title("Conversational Chain with RAG")

# Input form for user query
with st.form("query_form"):
    user_query = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit")

if submitted and user_query:
    # Set up the chain for retrieval and response
    setup = {"question": query_fetcher, "chat_history": history_fetcher | retriever | format_docs}
    _chain = setup | _prompt | chat_llm | StrOutputParser()

    # Prepare the chat history
    formatted_history = "\n".join([str(h) for h in st.session_state['chat_history']])

    # Invoke the chain
    response = _chain.invoke({"question": user_query, "chat_history": formatted_history})

    # Append to history and display
    query = f"user_question: {user_query}"
    response_str = f"ai_response: {response}"

    # Store the conversation in session state
    st.session_state['chat_history'].append((query, response_str))

    # Display the response
    st.write("AI Response:")
    st.write(response_str)

    # Display the conversation history
    st.write("Conversation History:")
    for msg in st.session_state['chat_history']:
        st.write(f"{msg[0]}\n{msg[1]}")

# Option to clear the chat history
if st.button("Clear History"):
    st.session_state['chat_history'] = []
    st.success("Chat history cleared!")
