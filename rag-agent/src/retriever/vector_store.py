from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

class VectorStoreRetriever:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 collection_name: str = 'elliott_wave_kb', persist_directory: str = './chroma_db'):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
    
    def load_and_index_pdf(self):
        # Check if vector store already exists with actual data
        chroma_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if os.path.exists(chroma_path):
            print(f"Loading existing vector store from {self.persist_directory}...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print("✓ Vector store loaded from disk!")
        else:
            print(f"Loading PDF from {self.pdf_path}...")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages. Splitting into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            splits = text_splitter.split_documents(documents)
            print(f"Creating vector store with {len(splits)} chunks...")
            
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            print(f"✓ Vector store created and saved to {self.persist_directory}!")
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        return self.retriever
