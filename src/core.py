import os
from typing import Any
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


class ChatAI:
    def __init__(self) -> ChatOllama:
        self.collection_name ='chroma_collection'
        self.persist_directory='chroma_db'
        self.data = "session/output.txt"
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.chat = ChatOllama(
            base_url="https://a089-34-142-146-166.ngrok-free.app",
            model="mistral",
            callback_manager=self.callback_manager)

    def add_data(self):
        loder = TextLoader(self.data, encoding="utf-8")
        document = loder.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator=".")
        texts = text_splitter.split_documents(document)
        db = Chroma.from_documents(texts, self.embedding_model, collection_name=self.collection_name, persist_directory=self.persist_directory)
        db.persist()
    
    
    def run_llm(self, query: str) -> Any:
        docsearch = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.chat,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )
        return qa({"query": query})

