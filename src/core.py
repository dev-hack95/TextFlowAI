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


def add_data():
    loder = TextLoader("session/output.txt", encoding="utf-8")
    document = loder.load()
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator=".")
    texts = text_splitter.split_documents(document)
    db = Chroma.from_documents(texts, embedding_model, collection_name="chroma_collection", persist_directory='chroma_db')
    db.persist()


def run_llm(query: str) -> Any:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    docsearch = Chroma(
        persist_directory='chroma_db',
        embedding_function=embedding_model,
        collection_name='chroma_collection'
    )
    chat = ChatOllama(
        base_url="https://a089-34-142-146-166.ngrok-free.app",
        model="mistral",
        callback_manager=callback_manager)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    return qa({"query": query})

