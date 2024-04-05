import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Config
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

if st.session_state.token == "":
    switch_page("app")

st.toast('Login Successful')

col1, col2 = st.columns([90, 10])

with col1:
    upload_video = st.file_uploader("Upload Video", type=["mp4"])
    #st.write(st.session_state.token)

with col2:
    if st.button("SignOut"):
        st.session_state.token = ""
        switch_page("app")