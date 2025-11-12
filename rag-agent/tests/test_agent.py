import unittest
from src.agent.rag_agent import RAGAgent
from src.retriever.vector_store import VectorStore
from src.generator.llm import LLM

class TestRAGAgent(unittest.TestCase):

    def setUp(self):
        self.vector_store = VectorStore()
        self.llm = LLM()
        self.rag_agent = RAGAgent()
        self.rag_agent.set_retriever(self.vector_store)
        self.rag_agent.set_generator(self.llm)

    def test_run_with_no_documents(self):
        response = self.rag_agent.run("What is the capital of France?")
        self.assertEqual(response, "No documents found.")

    def test_run_with_documents(self):
        self.vector_store.add_document("France is a country in Europe.")
        response = self.rag_agent.run("What is the capital of France?")
        self.assertIn("France", response)

    def test_set_retriever(self):
        new_vector_store = VectorStore()
        self.rag_agent.set_retriever(new_vector_store)
        self.assertEqual(self.rag_agent.retriever, new_vector_store)

    def test_set_generator(self):
        new_llm = LLM()
        self.rag_agent.set_generator(new_llm)
        self.assertEqual(self.rag_agent.generator, new_llm)

if __name__ == '__main__':
    unittest.main()