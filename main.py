from fastapi import FastAPI,Request
# from pydantic import BaseModel
import uvicorn
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import torch

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/hello")
def read_root():
    return {"Hello": "Hello"}

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
    model = AutoModelForSequenceClassification.from_pretrained("martin-ha/toxic-comment-model")
    return tokenizer,model

tokenizer,model = get_model()
pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text' in data:
        user_input = data['text']
        output = pipeline(user_input)
        response = {"Recieved Text": user_input,"Prediction": output[0]}
    else:
        response = {"Recieved Text": "No Text Found"}
    return response

if __name__ == "__main__":
    uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True)