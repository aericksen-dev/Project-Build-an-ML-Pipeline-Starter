import os, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import load_model, inference

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "model.pkl")
ENCODER_PATH = os.path.join(PROJECT_DIR, "model", "encoder.pkl")
LB_PATH = os.path.join(PROJECT_DIR, "model", "label_binarizer.pkl")

CAT_FEATURES = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

class Data(BaseModel):
    age:int; workclass:str; fnlgt:int; education:str; education_num:int=Field(...,alias="education-num")
    marital_status:str=Field(...,alias="marital-status")
    occupation:str; relationship:str; race:str; sex:str
    capital_gain:int=Field(...,alias="capital-gain")
    capital_loss:int=Field(...,alias="capital-loss")
    hours_per_week:int=Field(...,alias="hours-per-week")
    native_country:str=Field(...,alias="native-country")
    class Config: populate_by_name=True

model = load_model(MODEL_PATH); encoder = load_model(ENCODER_PATH); lb = load_model(LB_PATH)
app = FastAPI(title="Census Income Classifier")

@app.get("/")
def root(): return {"message":"Welcome to the Census Income Prediction API"}

@app.post("/predict")
def predict(d:Data):
    df = pd.DataFrame([d.model_dump(by_alias=True)])
    X,_,_,_ = process_data(df, CAT_FEATURES, label=None, training=False, encoder=encoder, lb=lb)
    pred = inference(model, X)[0]
    return {"prediction":int(pred),"label":">50K" if pred==1 else "<=50K"}
