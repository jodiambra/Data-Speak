
# 'Message' error
import os
import chainlit as cl
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


model_path = 'models\llama2_python_qa\llama-2-7b-finetuned-python-qa_tokenizer.gguf.fp16.bin'
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


TEMPLATE = """Question: {question}
Context: {context}
Answer: """

def custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=TEMPLATE,
                            input_variables=["context", "question"])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = model_path,
        model_type="llama",
        callback_manager=callback_manager,
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm



#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local('vectorstore', embeddings)
    llm = load_llm()
    qa_prompt = custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response



#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting your gen AI bot!...")
    await msg.send()
    msg.content = "Welcome to Demo Bot!. Ask your question here:"
    await msg.update()
    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=answer).send()