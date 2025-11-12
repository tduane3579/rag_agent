from src.utils.config import load_config
from src.retriever.vector_store import VectorStoreRetriever
from src.agent.rag_agent import ElliottWaveRAGAgent
import os

def main():
    print("=" * 50)
    print("Elliott Wave Expert RAG Agent")
    print("=" * 50 + "\n")
    config = load_config()
    if not config['anthropic_api_key']:
        print("Error: Edit .env file and add ANTHROPIC_API_KEY")
        return
    if not os.path.exists(config['pdf_path']):
        print(f"Error: Place all_ocr.pdf in data/documents/")
        return
    retriever_service = VectorStoreRetriever(pdf_path=config['pdf_path'], chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'], collection_name=config['collection_name'])
    retriever = retriever_service.load_and_index_pdf()
    agent = ElliottWaveRAGAgent(retriever=retriever, anthropic_api_key=config['anthropic_api_key'])
    agent.chat()

if __name__ == "__main__":
    main()
