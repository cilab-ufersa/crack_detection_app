# main.py
import streamlit as st
import home
import analyze_crack as other

params = st.query_params
page = params.get('page', ['home'])[0] 
img = params.get('image', [''])[0]

if st.query_params.get("page", "home") == "home":
    home.show()
elif st.query_params.get("page", "other") == "other":
    other.show()