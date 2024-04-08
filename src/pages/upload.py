import os
import json
import time
import requests
import warnings
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
from confluent_kafka import Consumer, Producer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import create_engine
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from langchain.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from streamlit_extras.switch_page_button import switch_page

# Config
load_dotenv(".env")
warnings.filterwarnings("ignore")

########################################## Streamlit Config #######################################################
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
if 'token' not in st.session_state:
    st.session_state.token = None

if st.session_state.token == "" or st.session_state.token == None:
    switch_page("app")

######################################### Config end ###############################################################

######################################### Sql Config ###############################################################
class DBLocalSession:
    def __init__(self) -> None:
        self.db_user = 'root'
        self.db_password = 'rushi12345'
        self.db_host = '192.168.29.7'
        self.db_name = 'user_service' 

    def LocalSession(self) -> sessionmaker:
        engine = create_engine(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        SessionLocal = sessionmaker(autoflush=False, bind=engine)
        db = SessionLocal()
        return db
    
session = DBLocalSession()
db = session.LocalSession()
Base = declarative_base()

class SchemaModel(Base):
    __tablename__ = 'video_data'
    id = Column(Integer, autoincrement=True, primary_key=True)
    email = Column(String)
    video = Column(String)
    audio = Column(String)
    text = Column(String)

######################################### Config end ###############################################################

######################################### Consumer Config ##########################################################
topic2 = "Kafkatopic2"
consumer = Consumer(
    {
        "bootstrap.servers": "192.168.29.7:9092",
        "group.id": "group2",
        "auto.offset.reset": "earliest",
    }
)

consumer.subscribe([topic2])
######################################## Config end ################################################################

col1, col2 = st.columns([90, 10])

def file_upload():
    output = None
    upload_video = st.file_uploader("Upload Video", type=["mp4"])
    if upload_video:
        filename = upload_video.name
        file_path = os.path.join('session/', filename)
        with open(file_path, 'wb') as f:
            f.write(upload_video.getvalue())

        url = 'http://192.168.29.100:8000/v1/upload'
        headers = {
            'Authorization': 'Bearer ' + st.session_state.token,
            'Content-Type': 'application/json'
        }
        data = {
            "email": st.session_state.email,
            "video": file_path,
            "audio": "",
            "text": "",
            "summary": ""
        }
        response = requests.post(url=url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            st.toast("Video Uploded Succesfully", icon="👍")
            st.toast("Please wait while processing the video", icon="🙏🏼")
            output = json.loads(response.text)
        else:
            st.error("Error Occured while uploding video")

        
    return output


def chatAI(video_text):
    with open("session/output.txt", 'w+') as file:
        file.write(video_text)

    st.header("🔗Chat With Video")
    chat = ChatOllama(
        base_url = "http://localhost:11434",
        model="mistral")
    memory = ConversationBufferWindowMemory(k=15)
    chain = ConversationChain(llm=chat, memory=memory)

    system_template = """You are an AI Assistant that helps users to solve their doubts from given context
    """
    human_template = """{question}"""
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_promt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    _ = chat_promt

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    loder = TextLoader("session/output.txt", encoding="utf-8")
    document = loder.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    texts = text_splitter.split_documents(document)
    docsearch = Chroma.from_documents(texts, embedding_model)
    qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=docsearch.as_retriever())

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
             {"role": "assistant", "content": "Hi, I'm a chatbot who help you to solve your doubts. How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt1 := st.chat_input(placeholder="How can i help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt1})
        st.chat_message("user").write(prompt1)

    prompt = qa({"query": prompt1})
    with st.spinner("Thinking..."):
        response = chain(prompt)['response']
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


with col2:
    if st.button("SignOut"):
        st.session_state.token = ""
        switch_page("app")

with col1:
    #data = file_upload()
    upload_video = st.file_uploader("Upload Video", type=["mp4"])
    if upload_video:
        filename = upload_video.name
        file_path = os.path.join('session/', filename)
        with open(file_path, 'wb') as f:
            f.write(upload_video.getvalue())

        url = 'http://192.168.29.100:8000/v1/upload'
        headers = {
            'Authorization': 'Bearer ' + st.session_state.token,
            'Content-Type': 'application/json'
        }
        data = {
            "email": st.session_state.email,
            "video": file_path,
            "audio": "",
            "text": "",
            "summary": ""
        }
        response = requests.post(url=url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            st.toast("Video Uploded Succesfully", icon="👍")
            st.toast("Please wait while processing the video", icon="🙏🏼")
        else:
            st.error("Error Occured while uploding video")

        consume = True
    
        while consume:
            msg = consumer.poll(1.0)
    
            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue
    
            data = json.loads(msg.value())
            video = data.get("video")
            output = data.get("text")
            print(output)

            if upload_video and output:
                consume = False
                print(output)
                chatAI(output)
        