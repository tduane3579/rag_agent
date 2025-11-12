from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStoreRetriever:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, collection_name: str = 'elliott_wave_kb'):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
    
    def load_and_index_pdf(self):
        print(f"Loading PDF from {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(documents)
        print(f"Creating vector store with {len(splits)} chunks...")
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embeddings, collection_name=self.collection_name)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Vector store created successfully!")
        return self.retriever
