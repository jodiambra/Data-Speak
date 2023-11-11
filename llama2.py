import streamlit as st
from PIL import Image
from streamlit.commands.page_config import Layout
from transformers import AutoTokenizer
import transformers
import torch
from transformers import pipeline
import requests
from streamlit_lottie import st_lottie
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import CTransformers
from transformers import AutoTokenizer
import sentence_transformers 
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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

#------------------------------------------------#

#create embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# load vectorstore from local
vectorstore = FAISS.load_local("vectorstore/", embeddings)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


#create llm
llm = CTransformers(
        model="llama-2-7b-finetuned-python-qa_tokenizer\llama-2-7b-finetuned-python-qa_tokenizer.gguf.fp16.bin",
        callback_manager=callback_manager,
        # verbose=True,
    )
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#-------------------------------------------------#
# Chat UI
chat_history = []

def generate_response(query):
  """Generates a response from the chatbot."""

  # Get the response from the conversational retrieval chain.
  response = chain({"question": query, "chat_history": chat_history})

  # Update the memory with the new conversation turn.
  chat_history.update(response)

  # Return the response.
  return response

# Create a text input field for the user to enter their query.
query = st.chat_input(placeholder="Enter your query:")

# Generate a response from the chatbot and display it to the user.
if query:
  response = generate_response(query)
  st.chat_message(response, role="bot")