import streamlit as st
from PIL import Image
import os
import time

st.title("Local Image Viewer")
filepath = "generated_images/output_image.png"

if os.path.exists(filepath):
    try:
        image = Image.open(filepath)
        st.image(image, caption="Auto-refreshed Image", width=1000)
    except OSError:
        st.warning("Invalid filepath. Please try again.")
    

    time.sleep(2)
    st.rerun()
else:
    st.warning("Image not found. Waiting for valid path...")