import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import xgboost
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Accept"]
)


class LoanData(BaseModel):
    loan_amnt: float
    issue_month: int
    issue_year: int
    term_mths: int
    annual_inc: float
    dti: float
    fico_range_low: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post('/predict')
def predict(data: LoanData):
    new_data = preprocess(data)

    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)
    if prediction[0] == 0:
        return {'prediction': int(prediction[0]),
                'probability': float(probability[0][0])}
    else:
        return {'prediction': int(prediction[0]),
                'probability': float(probability[0][1])}


def preprocess(input_data: LoanData):
    c_annual_inc = annual_inc_category(input_data.annual_inc)
    c_dti = dti_category(input_data.dti)
    c_fico_range_low = fico_range_low_category(input_data.fico_range_low)

    # create a df from the input data
    df = pd.DataFrame({
        'loan_amnt': [input_data.loan_amnt],
        'issue_year': [input_data.issue_year],
        'issue_month': [input_data.issue_month],
        'term_months': [input_data.term_mths],
        'annual_inc_category': [c_annual_inc],
        'dti_category': [c_dti],
        'fico_range_low_category': [c_fico_range_low]
    })

    return df


def annual_inc_category(annual_inc):
    if annual_inc <= 40000:
        return 1
    elif annual_inc <= 120000:
        return 2
    else:
        return 3


def dti_category(dti):
    if dti <= 10:
        return 1
    elif dti <= 20:
        return 2
    else:
        return 3


def fico_range_low_category(fico_range_low):
    if fico_range_low < 580:
        return 1
    elif fico_range_low < 670:
        return 2
    elif fico_range_low < 740:
        return 3
    elif fico_range_low < 800:
        return 4
    else:
        return 5


if __name__ == '__main__':
    uvicorn.run(app, port=1000, host='localhost')
