from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
import pandas as pd
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 🔹 Charger le modèle
with open("svm_model.pkl", "rb") as f:
    saved_package = pickle.load(f)

model = saved_package["model"]
saved_columns = saved_package["columns"]

# 🔹 Page principale
@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🔹 Route prédiction
@app.post("/predict")
def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    occupation: str = Form(...),
    education_level: str = Form(...),
    marital_status: str = Form(...),
    income: float = Form(...),
    credit_score: int = Form(...)
):
    # Créer DataFrame
    data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "education_level": education_level,
        "marital_status": marital_status,
        "income": income,
        "credit_score": credit_score
    }])

    # 🔹 One-Hot Encoding (comme training)
    data = pd.get_dummies(data)

    # 🔹 Aligner les colonnes
    data = data.reindex(columns=saved_columns, fill_value=0)

    # 🔹 Prédiction
    prediction = model.predict(data)[0]

    result = "✅ Loan Approved" if prediction == 0 else "❌ Loan Not Approved"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
