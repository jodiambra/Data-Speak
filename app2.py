# 'Message' object has no attribute 'replace'

import os
from langchain.llms import CTransformers
from dotenv import load_dotenv
import chainlit as cl
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA



model_path = 'models\llama2_python_qa\llama-2-7b-finetuned-python-qa_tokenizer.gguf.fp16.bin'
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = CTransformers(model = model_path, 
                    model_type="llama",
                    callback_manager=callback_manager,
                    max_new_tokens = 512,
                    temperature = 0.5
)
                    


template = """Question: {question}
Context : {context}
Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local('vectorstore', embeddings)
    llm_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(question, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()