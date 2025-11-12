from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ElliottWaveRAGAgent:
    def __init__(self, retriever, anthropic_api_key: str):
        self.retriever = retriever
        self.llm = ChatAnthropic(
            api_key=anthropic_api_key, 
            model="claude-3-haiku-20240307", 
            temperature=0.7
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Elliott Wave Theory with comprehensive knowledge from R.N. Elliott's original works.

Use the following context to answer the question with specific Elliott Wave principles.

Context: {context}"""),
            ("human", "{question}")
        ])
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> dict:
        answer = self.chain.invoke(question)
        docs = self.retriever.invoke(question)
        return {"answer": answer, "source_documents": docs}
    
    def chat(self):
        print("Elliott Wave RAG Agent initialized!")
        print("Ask me anything about Elliott Wave Theory (type 'quit' to exit)\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not question:
                continue
            print("\nThinking...\n")
            result = self.query(question)
            print(f"Agent: {result['answer']}\n")
            print(f"(Based on {len(result['source_documents'])} source chunks)\n")
