# Install libraries
!pip install -q langchain langchain-community langchain-core langchain-text-splitters
!pip install -q sentence-transformers faiss-cpu transformers pypdf


from google.colab import files
uploaded = files.upload()

# Rename PDF
!mv "/content/Annual-Report-FY-2023-24 (1) (1).pdf" /content/swiggy_report.pdf

# Install once
!pip install -q langchain langchain-community sentence-transformers faiss-cpu transformers pypdf


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


# Load PDF
loader = PyPDFLoader("/content/swiggy_report.pdf")
documents = loader.load()

print("Total pages:", len(documents))


# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)


# Remove noisy metadata sections
docs = [d for d in docs if "Registered Name" not in d.page_content]
docs = [d for d in docs if "Statutory Auditors" not in d.page_content]
docs = [d for d in docs if "secretarial@" not in d.page_content]


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector database
vectorstore = FAISS.from_documents(docs, embeddings)

print("Vector DB ready")


# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# Instruction model (better than GPT2)
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=40,
    do_sample=False
)

print("Model loaded")


while True:

    question = input("\nAsk a question about Swiggy report: ")

    if question.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:
"""

    output = generator(prompt)[0]["generated_text"]

    answer = output.replace(prompt, "").strip()

    print("\nAnswer:", answer)
