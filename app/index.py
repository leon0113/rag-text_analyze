from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "data" / "sample.pdf"

# pdf loader
loader = PyPDFLoader(str(pdf_path))

docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)


chunks = text_splitter.split_documents(documents=docs)


# embed the chunks locally with a free HuggingFace model (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


# create a qdrant vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="sample_collection",
    url="http://localhost:6333",
)