from fastapi import FastAPI
from Functions.classify import Classify
from Functions.ClassificationInsights import ClassificationInsights
from Functions.RegressionInsights import RegressionInsights
import pandas as pd



app = FastAPI()

@app.get("/")
async def root():
    data = pd.read_csv("Movie_classification.csv")
    return {'message':'Classification'} if Classify(data,'target') else {'message':'Regression'}

@app.get("/classificationInsights")
async def classify():
    data = pd.read_csv("Movie_classification.csv")
    return ClassificationInsights(data,"target")

@app.get("/regressionInsights")
async def regression():
    data = pd.read_csv("House_Price.csv")
    return RegressionInsights(data,"target")