import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml import model as m

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "census.csv")
CAT = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

def _mini():
    df = pd.read_csv(DATA_PATH).sample(n=500, random_state=0)
    tr, te = train_test_split(df, test_size=0.2, random_state=0)
    Xtr,ytr,enc,lb = process_data(tr, CAT, "salary", True)
    Xte,yte,_,_ = process_data(te, CAT, "salary", False, enc, lb)
    return Xtr,ytr,Xte,yte,enc,lb

def test_train_infer():
    Xtr,ytr,Xte,yte,enc,lb = _mini()
    clf = m.train_model(Xtr,ytr)
    assert m.inference(clf,Xte).shape[0]==Xte.shape[0]

def test_metrics():
    Xtr,ytr,Xte,yte,enc,lb = _mini()
    clf = m.train_model(Xtr,ytr)
    p,r,f1 = m.compute_model_metrics(yte,m.inference(clf,Xte))
    assert 0<=p<=1 and 0<=r<=1 and 0<=f1<=1

def test_save_load(tmp_path):
    Xtr,ytr,*_ = _mini()
    clf = m.train_model(Xtr,ytr)
    p = tmp_path/"m.pkl"; m.save_model(clf,str(p))
    clf2 = m.load_model(str(p))
    assert np.array_equal(clf.predict(Xtr[:5]), clf2.predict(Xtr[:5]))
