from fastapi import FastAPI
from schemas import InputSchema, OutputSchema
from predict import make_predict, make_batch_predict
from typing import List

app = FastAPI()

@app.get('/')
def index():
    return {"message": "Welcome to the Housing Price Prediction API"}

@app.post('/prediction', response_model=OutputSchema)
def predict_price(input_data: InputSchema):
    prediction = make_predict(input_data.model_dump())
    return OutputSchema(predicted_price=round(prediction,2))

@app.post('/batch_prediction',response_model=list[OutputSchema])
def batch_predict(user_data:List[InputSchema]):
    predictions=make_batch_predict([data.model_dump() for data in user_data])
    return [OutputSchema(predicted_price=round(pred,2)) for pred in predictions]