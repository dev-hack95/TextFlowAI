import os
from typing import Any
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
import pinecone

pinecone.init(
    api_key=os.environ["api_key"],
    environment=os.environ["environment"],
)

def add_data():
    loder = TextLoader("session/output.txt", encoding="utf-8")
    document = loder.load()
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator=".")
    texts = text_splitter.split_documents(document)
    Pinecone.from_documents(texts, embedding_model)


def run_llm(query: str) -> Any:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    docsearch = Pinecone.from_existing_index(
        index_name="testindex2", embedding=embeddings
    )
    chat = ChatOllama(
        base_url="https://fbc6-104-198-103-21.ngrok-free.app",
        model="mistral:7b",
        callback_manager=callback_manager)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    return qa({"query": query})

