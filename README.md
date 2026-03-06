# RAG Question Answering System – Swiggy Annual Report

## Objective
Build a Retrieval-Augmented Generation (RAG) system that answers questions based strictly on the Swiggy Annual Report.

#

## Technologies Used
- Python
- LangChain
- FAISS Vector Database
- Sentence Transformers (all-MiniLM-L6-v2)
- HuggingFace Transformers
- TinyLlama LLM

## System Architecture
1. Load Swiggy Annual Report PDF
2. Split document into text chunks
3. Generate embeddings using sentence-transformers
4. Store embeddings in FAISS vector database
5. Retrieve relevant document chunks
6. Pass context to LLM
7. Generate final answer

## Example Query
Question:
What is Swiggy?

Answer:
Swiggy is a hyperlocal commerce platform that provides food delivery and convenience services across multiple cities in India.

## How to Run
1. Install dependencies

pip install langchain langchain-community sentence-transformers faiss-cpu transformers pypdf

2. Run the script

python rag_swiggy.py

3. Ask questions in the CLI
