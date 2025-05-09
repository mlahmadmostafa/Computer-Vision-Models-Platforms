import streamlit as st

st.title("Computer Vision Platform")

name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! 👋")


