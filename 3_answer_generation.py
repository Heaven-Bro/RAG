from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# Load the SAME embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load ChromaDB
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# User query
query = "How much did Microsoft pay to acquire GitHub?"

# Retrieve relevant documents
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

# Display retrieved context
print("--- Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:")
    print(doc.page_content)
    print()


# Combine query + retrieved documents
combined_input = f"""
Based on the following documents, answer the question.

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Instructions:
- Answer using only the information from the documents.
- Do not make up information.
- If the answer cannot be found in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
- Give a clear and concise answer.
"""


# Gemini LLM
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# Messages
messages = [
    SystemMessage(
        content="You are a helpful RAG assistant. Answer only from the provided context."
    ),
    HumanMessage(content=combined_input),
]


# Generate answer
result = model.invoke(messages)


# Display final answer
print("\n--- Generated Response ---")
print(result.content[0]["text"])

