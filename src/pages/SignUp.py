import time
import json
import requests
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Config
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

def signup():
    with st.form(key="signup", clear_on_submit=True):
        st.subheader(':green[Sign Up]')
        firstname = st.text_input("FirstName", placeholder="First Name")
        lastname = st.text_input("LastName", placeholder="Last Name")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", placeholder="Enter Password", type='password')
        confirm_password = st.text_input("Confirm Password", placeholder="Confirm Password", type='password')
        if st.form_submit_button("Create Account"):
            inputs = {
                "first_name": firstname,
                "last_name": lastname,
                "email": email,
                "password": password,
                "confirm_password": confirm_password,
                "admin": True
            }
            response = requests.post(url="http://192.168.29.100:8000/v1/signup", data=json.dumps(inputs))
            if response.status_code == 200:
                st.success("Account is created succesfully")
                st.toast("Redirect to Login Page......")
                time.sleep(2)
                switch_page("app")
            else:
                st.error("User name and password doesn't match")

col1, col2= st.columns(2)

with col1:
    st.image("assets/login_page.jpg")

with col2:
    signup()
    st.markdown('<a href="/" target="_self">Sign In</a>', unsafe_allow_html=True)