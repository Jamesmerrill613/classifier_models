from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from knn import predict as knn_predict
from rfc import predict as rfc_predict
from svm import predict as svm_predict
from lrc import predict as lrc_predict

app = FastAPI()

class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.put("/knn")
def predict_knn(item: Item):
    target = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width])
    return { "predict": {knn_predict(target)} }

@app.put("/rfc")
def predict_rfc(item: Item):
    target = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width])
    return { "predict": {rfc_predict(target)} }

@app.put("/svm")
def predict_svm(item: Item):
    target = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width])
    return { "predict": {svm_predict(target)} }

@app.put("/lrc")
def predict_lrc(item: Item):
    target = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width])
    return { "predict": {lrc_predict(target)} }

@app.put("/all")
def predict_all(item: Item):
    target = np.array([item.sepal_length, item.sepal_width, item.petal_length, item.petal_width])
    return { "knn": {knn_predict(target)},
             "rfc": {rfc_predict(target)},
             "svm": {svm_predict(target)},
             "lrc": {lrc_predict(target)} }