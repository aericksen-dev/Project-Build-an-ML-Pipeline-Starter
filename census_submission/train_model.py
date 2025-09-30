from __future__ import annotations
import os, json, pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference, performance_on_categorical_slice, save_model, train_model

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "census.csv")
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "model")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoder.pkl")
LB_PATH = os.path.join(ARTIFACT_DIR, "label_binarizer.pkl")
SLICE_OUTPUT_PATH = os.path.join(PROJECT_DIR, "slice_output.txt")
METRICS_JSON_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

CAT_FEATURES = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]

def main():
    data = pd.read_csv(DATA_PATH)
    train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    X_train, y_train, enc, lb = process_data(train, CAT_FEATURES, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, CAT_FEATURES, label="salary", training=False, encoder=enc, lb=lb)
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, f1 = compute_model_metrics(y_test, preds)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    save_model(model, MODEL_PATH); save_model(enc, ENCODER_PATH); save_model(lb, LB_PATH)
    with open(METRICS_JSON_PATH, "w") as f: json.dump({"precision":p,"recall":r,"f1":f1}, f, indent=2)
    slices = performance_on_categorical_slice(test, "education", enc, lb, model, CAT_FEATURES)
    with open(SLICE_OUTPUT_PATH, "w") as f:
        f.write("Slice performance by education\n")
        for v,(pp,rr,ff) in slices.items():
            f.write(f"education={v}: precision={pp:.4f}, recall={rr:.4f}, f1={ff:.4f}\n")
    print("Done. Artifacts in ./model, slices in slice_output.txt")

if __name__ == "__main__":
    main()
