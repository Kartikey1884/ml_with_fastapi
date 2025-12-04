from fastapi import FastAPI
from schemas import InputSchema, OutputSchema
from predict import make_predict

app = FastAPI()

@app.get('/')
def index():
    return {"message": "Welcome to the Housing Price Prediction API"}

@app.post('/prediction', response_model=OutputSchema)
def predict_price(input_data: InputSchema):
    prediction = make_predict(input_data.model_dump())
    return OutputSchema(predicted_price=round(prediction,2))
