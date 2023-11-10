# Data-Speak

## Company

DataSpeak (https://dataspeak.co/) utilizes Predictive Analytics and Machine Learning capabilities to assist organizations in gaining a competitive edge in their respective markets through actionable intelligence derived from existing data sources, including sales and marketing data, weather data, and supply chain data. The solutions use statistical and mathematical analysis techniques to uncover hidden data patterns to enhance business decisions and increase profits. They have become one of the industry's largest providers of predictive analytics solutions through a combination of in-house software development and an extensive reseller network. In addition, they provide bespoke solutions to a wide range of industries worldwide.

## Task

The task is to train a machine learning model that can automatically generate answers to written questions a user inputs. For this purpose, we will train a model with questions and answers using the Python Questions from Stack Overflow dataset. 

The input to our model would be the written questions (in English), and the model produces the top answers.Once an accurate model was trained, an application for a user interface was built where the user could input the question in a text box, and the top answers appear.


## Data Processing

We used beautifulsoup to process the html in the text. We combined the title and body into a column called full question. The best answers for each question were picked based on answer score. The unnecessary columns were removed, so that only the questions and answers were remaining. 

## Modeling

### LLAMA 2

We used Llama 2, a new model designed by Meta AI. This version was released in July of 2023, and is one of the most comprehensive models available. We chose this model due to its performance, and because it is open source. 

We implemented RAG (Retreival Augmented Generation), as it allows us to interact with internal knowledge bases and external customer-facing documentation. The main benefit of RAG is the access to the model's sources, ensuring that the answers given can be checked for accuracy, and therefore trusted. 

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/0caa086b-f564-4682-936c-c193844fd60c)


### Langchain
Langchain is a framework that helps to manage large language models. It revolutionizes chatbots, question/answer, and summarization by chaining components from multiple modules. Langchain was used to tie together our model, prompts, indexes, chains, and memory stored as a vectorstore. Langchain also made it easy to incorporate our model into our website.  

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/23e26c4e-0f4f-4b46-896b-4b8a200fd078)

### Model

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/cd8213f8-5b42-415b-a575-fd69667139b2)

Our model implemented a conversational retrieval chain, which retrieved our answers from our embeddings, saved in a vectorstore. We also implemented conversation buffer memory, so we could ask follow up questions, and the model would keep previous responses in memory. Finally, we allowed the model to return source documents upon request, so users area able to verify the responses. 

### Deployment

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/20be101a-568a-4e3c-9d07-14b11466a47f)


Our model was pushed to [HuggingFace](https://huggingface.co/jodiambra/llama-2-7b-finetuned-python-qa_tokenizer/tree/main), where it can be deployed using Amazon Sagemaker, Azure ML, or another paid deployment. 


[Notebook](llama.ipynb) used to run the model. 

## Webpages

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/ceee890b-27e4-4e23-92c0-8102c579024a)

![image](https://github.com/jodiambra/Data-Speak/assets/115895428/b60161c3-5f67-486b-9aaf-4ce0bd90ff69)
