# Time out error

import chainlit as cl
from langchain.llms import CTransformers
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os 



os.environ['API_KEY'] = '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'


model_id = 'jodiambra/llama-2-7b-finetuned-python-qa_tokenizer-GGML'

llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

template = """

You are an AI assistant that provides helpful answers to user queries.

{question}

"""
prompt = PromptTemplate(template=template, input_variables=['question'])




# How to react when connection is established
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session

    prompt = PromptTemplate(template=template, input_variables=["question"])

    #Invoked every time message is sent
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Sets the chain in the user session
    # user_session is a dictionary with set and get methods
    # key : "llm_chain" , value :  llm_chain
    cl.user_session.set("llm_chain", llm_chain)




#  How to react each time a user sends a message.
#  Then, we send back the answer to the UI with the Message class.
@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session by Passing the key
    llm_chain = cl.user_session.get("llm_chain")

    # Call the chain asynchronously - Use await keyword to call asynchronous function
    # Components communicating : Chainlit UI and langchain agent
    # When running your LangChain agent, it is essential to provide the appropriate callback handler.
    # acall is a function in Chainlit LangChain that allows you to call an asynchronous function with a message as input.
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    answer = res["result"]
    print("Result : ", answer)

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.

    # Use Message class to send, stream, edit or remove messages in the chatbot user interface.
    await cl.Message(content=answer).send()