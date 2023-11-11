import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import CTransformers



TEMPLATE = """Question: {question}

Answer: """


def get_llm_chain(model_name: str) -> LLMChain:
    """Get LLMChain with LlamaCpp as LLM.

    Args:
        model_name (str): Name of the Llama model.

    Returns:
        LLMChain: LLMChain with a given model.
    """
    prompt = PromptTemplate(template=TEMPLATE, input_variables=["question"])
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = CTransformers(
        model=model_name,
        callback_manager=callback_manager,
        # verbose=True,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


def main() -> None:
    """Langchain + Streamlit + Llama = Bot 🤖"""

    st.set_page_config(page_title="Python Stack Overflow Chatbot 🤖", layout="wide")
    st.title("PyBot 🤖")
    st.subheader("Langchain 🦜️⛓️ + Streamlit 👑 + Llama 🦙 + Python	:snake: ")

    llm_chain = get_llm_chain(model_name="llama-2-7b-finetuned-python-qa_tokenizer\llama-2-7b-finetuned-python-qa_tokenizer.gguf.fp16.bin")

    question = st.chat_input("What do you want to ask?")
    

    if question:
        with st.spinner('Generating response...'):
            output = llm_chain.run(question)
        st.chat_message(output)


if __name__ == "__main__":
    main()