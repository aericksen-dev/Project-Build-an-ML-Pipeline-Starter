import requests
BASE="http://127.0.0.1:8000"
def main():
    print("GET / ->", requests.get(BASE+"/").status_code)
    sample={"age":39,"workclass":"State-gov","fnlgt":77516,"education":"Bachelors","education-num":13,
        "marital-status":"Never-married","occupation":"Adm-clerical","relationship":"Not-in-family",
        "race":"White","sex":"Male","capital-gain":2174,"capital-loss":0,"hours-per-week":40,
        "native-country":"United-States"}
    r = requests.post(BASE+"/predict", json=sample)
    print("POST /predict ->", r.status_code, r.json())
if __name__=="__main__": main()
