#import packages
import streamlit as st
import pandas as pd
import plotly_express as px
from PIL import Image
from streamlit.commands.page_config import Layout
import requests
from streamlit_lottie import st_lottie

#----------------------------#
# Upgrade streamlit library
# pip install --upgrade streamlit

#-----------------------------#
# Page layout
icon = Image.open('images/Python.png')

st.set_page_config(page_title='Python Stack Overflow',
                   page_icon=icon,
                   layout='wide',
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title('Python Stack Overflow Questions and Answers')

# lottie Animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()



stack = load_lottieurl('https://lottie.host/3edc446c-ec1d-40cd-882e-f7a8f308a5f7/6FSDe4sodo.json')

python = load_lottieurl('https://lottie.host/4f208107-d2cf-47cf-92aa-9316c8c9ed8b/DkRDv2zxGM.json')

col1, col2 = st.columns(2)

with col1:
    st_lottie(python, height=300, width=450, quality='high', speed=0.35)
with col2:
    st_lottie(stack, height=300, width=450, quality='high', speed=1)



st.text_input('Question', placeholder='Type Questions Here')

st.text_area('Answers')