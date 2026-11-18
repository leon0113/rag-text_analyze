import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# load the API key etc. from the .env file in the project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

router = APIRouter(prefix="/query", tags=["query"])

# same embedding model we used in index.py (must match!)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# connect to the collection we already created
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="sample_collection",
    url="http://localhost:6333",
)

# Tokens.bd works like OpenAI, so we use the openai library with a different url
client = OpenAI(
    base_url=os.getenv("TOKENS_BASE_URL"),
    api_key=os.getenv("TOKENS_API_KEY"),
)


@router.post("")
def query_endpoint(question: str):
    # 1. find the 4 most similar chunks to the question
    results = vector_db.similarity_search(query=question, k=4)

    # 2. join the chunks into one text
    context = ""
    for chunk in results:
        context += chunk.page_content + "\n\n"

    # 3. ask the LLM to answer using only that text
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from the provided context. "
                'If the answer is not in the context, say "I don\'t know based on the documents." '
                "Keep answers concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    # 4. send the answer back
    answer = response.choices[0].message.content
    return {
        "question": question,
        "answer": answer
        }
