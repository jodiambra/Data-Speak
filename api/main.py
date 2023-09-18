# Importing necessary libraries
## pydantic==1.8.2
## fastapi==0.68.1
## uvicorn==0.15.0

import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initializing the fast API server
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading up the trained model
model = pickle.load(open('models/full_model.pkl', 'rb'))


# Defining the model input types
class Candidate(BaseModel):
    question: str
   


# Setting up the home route
@app.get("/")
def read_root():
    return {"data": "Welcome to the Stack Overflow Question and Answer Interface"}


# Setting up the prediction route
@app.post("/prediction/")
async def get_predict(data: Candidate):
    sample = [[
        data.question,
    ]]
    answer = model.predict(sample).tolist()[0]
    probability = model.predict_proba(sample).tolist()[0]
    prob = probability[1]*100 
    return {
        "data": {
            'prediction': answer,
            'probability': prob,
            'interpretation': 'Answers to the question, ranked by best answer'
        }
    }


# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')