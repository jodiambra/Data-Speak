import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers



st.title('🦜🔗 Quickstart App')

TEMPLATE = """Question: {question}

Answer: """

prompt = PromptTemplate(template=TEMPLATE, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def generate_response(input_text):
    llm = CTransformers(
        model="llama-2-7b-finetuned-python-qa_tokenizer\llama-2-7b-finetuned-python-qa_tokenizer.gguf.fp16.bin",
        callback_manager=callback_manager,
        # verbose=True,
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    st.info(llm_chain(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'how do you make a list in python?')
    submitted = st.form_submit_button('Submit')

    generate_response(text)


