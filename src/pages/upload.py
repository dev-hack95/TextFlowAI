import os
import json
import time
import requests
import warnings
import pinecone
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import create_engine
from langchain.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from streamlit_extras.switch_page_button import switch_page

# Config
load_dotenv(".env")
warnings.filterwarnings("ignore")
pinecone.init(api_key=os.environ.get("api_key"), environment=os.environ.get("environment"))

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
        self.db_user = os.environ.get("db_user")
        self.db_password = os.environ.get("db_password")
        self.db_host = os.environ.get("db_host")
        self.db_name = os.environ.get("db_name")

    def LocalSession(self) -> sessionmaker:
        engine = create_engine(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")
        SessionLocal = sessionmaker(autoflush=False, bind=engine)
        db = SessionLocal()
        return db
    
session = DBLocalSession()
db = session.LocalSession()

######################################### Config end ###############################################################

col1, col2 = st.columns([90, 10])

def chatAI(video_text: str):
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

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    #loder = TextLoader('medium.txt', encoding="utf-8")
    document = video_text
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    texts = text_splitter.split_documents(document)
    docsearch = Pinecone.from_documents(texts, embedding_model, index_name = "testindex1")
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

    prompt = chat_promt.format_prompt(question=qa({"query": prompt1}),).to_messages()
    with st.spinner("Thinking..."):
        response = chain(prompt)['response']
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


with col1:
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

        output_data = json.loads(response.text)
        while True:
            res_text = db.execute(text(f"SELECT text FROM video_data WHERE video = {output_data['video']}")).fetchone()
            if res_text:
                print(res_text)
                break
            time.sleep(5)

        if res_text != "":
            chatAI(res_text)

with col2:
    if st.button("SignOut"):
        st.session_state.token = ""
        switch_page("app")