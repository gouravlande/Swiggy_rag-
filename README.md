# RAG Question Answering System – Swiggy Annual Report

## Objective

Build a Retrieval-Augmented Generation (RAG) system that answers user questions strictly based on the Swiggy Annual Report.

---

## Technologies Used

* Python
* LangChain
* FAISS Vector Database
* Sentence Transformers (all-MiniLM-L6-v2)
* HuggingFace Transformers
* TinyLlama LLM

---

## System Architecture

1. Load the Swiggy Annual Report PDF
2. Split the document into smaller text chunks
3. Generate embeddings using Sentence Transformers
4. Store embeddings in a FAISS vector database
5. Retrieve the most relevant chunks for a question
6. Pass the retrieved context to the LLM
7. Generate the final answer based on the retrieved context

---

## Architecture Diagram

```
Swiggy PDF
   ↓
Document Loader
   ↓
Text Chunking
   ↓
Embeddings (MiniLM)
   ↓
FAISS Vector Database
   ↓
Retriever
   ↓
LLM
   ↓
Answer
```

---

## Workflow

```
Load PDF
   ↓
Chunk Text
   ↓
Create Embeddings
   ↓
Store in FAISS
   ↓
Retrieve Relevant Chunks
   ↓
Generate Answer
   ↓
CLI Question Input
```

---

## Example Query

Question:
What is Swiggy?

Answer:
Swiggy is a hyperlocal commerce platform that provides food delivery and convenience services across multiple cities in India.

---

## How to Run

### 1. Install Dependencies

```
pip install langchain langchain-community sentence-transformers faiss-cpu transformers pypdf
```

### 2. Run the Script

```
python rag_swiggy.py
```

### 3. Ask Questions

```
Ask a question about Swiggy report: What is Instamart?
```

---

## Project Structure

```
RAG-Swiggy-Report
│
├── rag_swiggy.py
├── README.md
├── requirements.txt
└── swiggy_report.pdf
```
