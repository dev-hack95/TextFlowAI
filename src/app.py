import json
import requests
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Config
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

token = None

def signin() -> str:
    global token
    with st.form(key="signin", clear_on_submit=True):
        st.subheader(':green[Sign In]')
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", placeholder="Enter Password", type='password')
        if st.form_submit_button("Login"):
            inputs = {
                "email": email,
                "password": password
            }
            response = requests.post(url="http://localhost:8000/v1/signin", data=json.dumps(inputs))
            if response.status_code == 200:
                data = json.loads(response.text)
                token = data['model']
                st.session_state.token = token
                switch_page("upload")
            else:
                st.error("User name and password doesn't match")

col1, col2= st.columns([50, 50])

with col1:
    st.image("assets/login_page.jpg")

with col2:
    signin()
    st.markdown('<a href="/SignUp" target="_self">Create New Account</a>', unsafe_allow_html=True)

    
