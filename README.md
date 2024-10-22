# AI-DocRetriever

This project is an implementation of a Retrieval-Augmented Generation (RAG) System that processes HTML documents by splitting them based on headers, chunking the text, and storing the chunks in a Qdrant Vector Store. It also utilizes OpenAI’s language model for conversational responses based on the context from the retrieved documents. The project uses the LangChain framework and various components like embedding models and text splitters.

Features
Load and split HTML files based on header tags.
Chunk the content into smaller segments for efficient document retrieval.
Store the document chunks in a Qdrant Vector Store.
Perform similarity-based searches to retrieve relevant chunks.
Use OpenAI’s GPT model to answer user queries based on retrieved content.
Installation
To run this project locally, follow the steps below.

Requirements
Python 3.8+
Qdrant
OpenAI API Key
Required Python packages (installed in the next step)
