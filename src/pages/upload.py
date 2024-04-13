import os
import json
import requests
import warnings
import streamlit as st
from dotenv import load_dotenv
from confluent_kafka import Consumer
from streamlit_extras.switch_page_button import switch_page
from core import add_data, run_llm

# Config
load_dotenv(".env")
warnings.filterwarnings("ignore")

########################################## Streamlit Config #######################################################
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
if 'token' not in st.session_state:
    st.session_state.token = None

if st.session_state.token == "" or st.session_state.token is None:
    switch_page("app")

######################################### Config end ###############################################################

consumer = Consumer({
                "bootstrap.servers": "192.168.29.7:9092",
                "group.id": "group2",
                "auto.offset.reset": "earliest",
            })

consumer.subscribe(["Kafkatopic2"])

consume = True

tab1, tab2 = st.tabs(["Upload", "Chat"])

with tab1:
    col1, col2 = st.columns([90, 10])
    with col1:
        with st.form(key="upload"):
            upload_video = st.file_uploader("Upload Video", type=["mp4"])
            if st.form_submit_button("Upload"):
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
                    st.toast("Video Uploaded Successfully 🎉")
                    st.toast("Please wait while processing the video 🙏🏼")
                else:
                    st.error("Error occurred while uploading video ❌")
        
                while consume:
                    message = consumer.poll(1.0)
        
                    if message is None:
                        continue
                    if message.error():
                        print("Consumer error: {}".format(message.error()))
                        continue
        
                    data = json.loads(message.value())
                    output_text = data.get("text")
        
                    if upload_video is not None and output_text is not None:
                        consume = False
                        with open("session/output.txt", 'w+') as file:
                            file.write(output_text)
                        st.toast("Your ready to chat")
                    else:
                        consume = True
                print(consume)

    with col2:
        if st.button("SignOut"):
            st.session_state.token = ""
            switch_page("app")

with tab2:
        st.header("🔗AI Bot")
        add_data()

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot who helps you in gym training and dietetics. How can I help you?"}
            ]
        
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        if prompt1 := st.chat_input(placeholder="How can I help you?"):
            st.session_state.messages.append({"role": "user", "content": prompt1})
            st.chat_message("user").write(prompt1)
        
            with st.spinner("Thinking..."):
                response = run_llm(prompt1)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)