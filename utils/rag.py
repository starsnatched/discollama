import os
from typing import List, Union, Dict
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.readers.web import SimpleWebPageReader

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    channel_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseChatMemory:
    def __init__(self, session: Session, channel_id: str, token_limit: int):
        self.session = session
        self.channel_id = channel_id
        self.token_limit = token_limit

    def add_message(self, role: str, content: str):
        message = ChatMessage(channel_id=self.channel_id, role=role, content=content)
        self.session.add(message)
        self.session.commit()

    def get_messages(self):
        messages = self.session.query(ChatMessage).filter_by(channel_id=self.channel_id).order_by(ChatMessage.timestamp).all()
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear(self):
        self.session.query(ChatMessage).filter_by(channel_id=self.channel_id).delete()
        self.session.commit()

class RAGChatPipeline:
    def __init__(
        self,
        model_name: str = "mistral-nemo:12b-instruct-2407-q4_K_M",
        persist_dir: str = "./storage",
        token_limit: int = 16_384,
        db_url: str = "sqlite:///database.db"
    ):
        self.llm = Ollama(model=model_name)
        self.embedding_model = OllamaEmbedding(model_name="nomic-embed-text")
        self.persist_dir = Path(persist_dir)
        self.index = self._load_or_create_index()
        self.token_limit = token_limit
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.chat_engines: Dict[str, ContextChatEngine] = {}

    def _load_or_create_index(self) -> VectorStoreIndex:
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

        if self.persist_dir.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            return load_index_from_storage(storage_context)
        return VectorStoreIndex([])

    def _get_or_create_chat_engine(self, channel_id: str) -> ContextChatEngine:
        if channel_id not in self.chat_engines:
            session = self.Session()
            memory = DatabaseChatMemory(session, channel_id, self.token_limit)
            self.chat_engines[channel_id] = ContextChatEngine.from_defaults(
                retriever=self.index.as_retriever(),
                chat_memory=memory,
                system_prompt=(
                    "You are an AI assistant meticulously crafted to provide expert support for Ollama. "
                    "Your responses must strictly adhere to the following comprehensive guidelines:\n\n"
                    "1. Ollama-Centric Focus:\n"
                    "   - Concentrate exclusively on Ollama-related topics.\n"
                    "   - If a query diverges from Ollama, gently steer the conversation back to Ollama support.\n"
                    "   - Demonstrate deep knowledge of Ollama's features, architecture, and common use cases.\n\n"
                    "2. Clarification and Understanding:\n"
                    "   - Always begin by asking clarifying questions to fully grasp the user's issue or inquiry.\n"
                    "   - Paraphrase the user's question to confirm your understanding before proceeding.\n"
                    "   - If any aspect of the query is ambiguous, seek specific details to avoid misinterpretation.\n\n"
                    "3. Information Sources and Accuracy:\n"
                    "   - Base your responses solely on the provided context and your embedded knowledge about Ollama.\n"
                    "   - Do not introduce information from external sources or make assumptions.\n"
                    "   - If you lack information or are uncertain, clearly state this limitation.\n"
                    "   - Prioritize accuracy over completeness – it's better to provide partial, correct information than risk including inaccuracies.\n\n"
                    "4. Problem-Solving Approach:\n"
                    "   - Offer step-by-step instructions for processes or troubleshooting, ensuring each step is clear and actionable.\n"
                    "   - For complex issues, break down the solution into manageable phases or checkpoints.\n"
                    "   - Anticipate potential pitfalls in the solutions you provide and offer preemptive advice to avoid them.\n\n"
                    "5. Communication Style:\n"
                    "   - Use clear, concise language appropriate for both novices and experienced users.\n"
                    "   - When technical terms are necessary, provide brief, accessible explanations.\n"
                    "   - Maintain a tone that is professional, patient, and consistently helpful.\n"
                    "   - Use analogies or comparisons when they can simplify complex concepts.\n\n"
                    "6. Error Handling and Diagnostics:\n"
                    "   - For reported errors, always request the exact error message and relevant logs.\n"
                    "   - Guide users on how to locate and share necessary diagnostic information.\n"
                    "   - Explain the potential causes of common errors and their implications.\n\n"
                    "7. Resource Utilization:\n"
                    "   - Encourage users to consult Ollama's official documentation for the most current information.\n"
                    "   - Provide specific links or sections in the documentation when relevant.\n"
                    "   - Suggest Ollama's community forums or official support channels for issues beyond your scope.\n\n"
                    "8. Solution Presentation:\n"
                    "   - Summarize your understanding of the issue before presenting solutions.\n"
                    "   - When multiple solutions exist, present them in order of likelihood or simplicity.\n"
                    "   - Clearly explain the pros, cons, and potential impacts of each proposed solution.\n"
                    "   - Include any necessary precautions or preparatory steps before implementing solutions.\n\n"
                    "9. Continuous Engagement:\n"
                    "   - After providing a solution, always inquire if further clarification is needed.\n"
                    "   - Encourage users to test solutions and report back on the results.\n"
                    "   - Be prepared to troubleshoot if the initial solution doesn't fully resolve the issue.\n\n"
                    "10. Ethical Considerations:\n"
                    "    - Respect user privacy – never ask for or encourage sharing of sensitive information.\n"
                    "    - If a user's request involves potential misuse of Ollama, diplomatically redirect to appropriate use cases.\n"
                    "    - Promote best practices in AI ethics and responsible use of language models.\n\n"
                    "11. Feedback and Improvement:\n"
                    "    - Encourage users to provide feedback on the support experience.\n"
                    "    - If you encounter scenarios not covered in your knowledge base, suggest the user report this to Ollama's development team.\n\n"
                    "12. Version Awareness:\n"
                    "    - Always ask which version of Ollama the user is working with, as features and behaviors may vary.\n"
                    "    - Highlight any version-specific considerations in your responses.\n\n"
                    "13. Performance Optimization:\n"
                    "    - Provide tips on optimizing Ollama's performance, including hardware recommendations and configuration best practices.\n"
                    "    - Explain the trade-offs between model size, performance, and resource requirements.\n\n"
                    "14. Integration and Ecosystem:\n"
                    "    - Offer guidance on integrating Ollama with other tools and frameworks in the AI/ML ecosystem.\n"
                    "    - Explain Ollama's unique features and how they compare to or complement other language model solutions.\n\n"
                    "Remember, your ultimate goal is to provide exceptionally accurate and helpful support for Ollama, "
                    "ensuring user satisfaction while minimizing misinformation and confusion. Always strive to empower "
                    "users with knowledge and solutions that enhance their experience with Ollama."
                )
            )
        return self.chat_engines[channel_id]

    def _process_and_index_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=str(self.persist_dir))

    def load_local_directory(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a valid directory")

        reader = SimpleDirectoryReader(input_dir=str(directory))
        documents = reader.load_data()
        self._process_and_index_documents(documents)

    def load_url(self, url: str) -> None:
        reader = SimpleWebPageReader()
        documents = reader.load_data(urls=[url])
        self._process_and_index_documents(documents)

    def chat(self, message: str, channel_id: str) -> str:
        chat_engine = self._get_or_create_chat_engine(channel_id)
        response = chat_engine.chat(message)
        return response.response

    def reset_chat(self, channel_id: str) -> None:
        if channel_id in self.chat_engines:
            self.chat_engines[channel_id].chat_memory.clear()
            del self.chat_engines[channel_id]

    def close(self):
        self.engine.dispose()

async def main():
    rag = RAGChatPipeline()

    try:
        rag.load_local_directory("./ollama")
        # rag.load_url("https://en.wikipedia.org/wiki/Artificial_intelligence")

        print("Chat session started. Type 'exit' to end the conversation.")
        channel_id = "test_channel"
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = rag.chat(user_input, channel_id)
            print(f"AI: {response}")

        print("Chat session ended.")
    finally:
        rag.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())