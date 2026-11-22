# -*- coding: utf-8 -*-
"""Service dự đoán đột quỵ: nhận JSON từ Go, trả JSON về cho Go.

Chạy (trong thư mục train/):  .venv\Scripts\python.exe -m uvicorn api:app --port 8000
Thử nhanh không cần Go:       mở http://localhost:8000/docs
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
bundle = joblib.load(BASE_DIR / "model.joblib")

pipeline = bundle["pipeline"]
calibrator = bundle["calibrator"]
threshold = bundle["threshold"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
max_age_trained = bundle["max_age_trained"]

app = FastAPI()


# Health check: dùng cho Render và UptimeRobot (GET + HEAD) để giữ service luôn thức
@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


def add_features(d):
    d = d.copy()
    d['age_group'] = pd.cut(d['age'], bins=[0, 40, 55, 65, 75, 120],
                            labels=[0, 1, 2, 3, 4]).astype(int)
    d['is_senior'] = (d['age'] >= 60).astype(int)
    d['bmi_obese'] = (d['bmi'] >= 30).astype(int)
    d['glucose_high'] = (d['avg_glucose_level'] >= 140).astype(int)
    d['comorbidity'] = d['hypertension'] + d['heart_disease']
    d['smoking_binary'] = d['smoking_status'].isin(['smokes', 'formerly smoked']).astype(int)
    d['risk_score'] = (d['age'] / 10
                       + d['hypertension'] * 3
                       + d['heart_disease'] * 4
                       + d['glucose_high'] * 2
                       + d['bmi_obese'] * 1
                       + d['smoking_binary'] * 2)
    return d


def logit(prob):
    prob = np.clip(prob, 1e-4, 1 - 1e-4)
    return np.log(prob / (1 - prob)).reshape(-1, 1)


class Patient(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    age: float = Field(gt=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float = Field(gt=0)
    bmi: float = Field(gt=0)
    smoking_status: Literal['never smoked', 'formerly smoked', 'smokes', 'Unknown']


@app.post("/predict")
def predict(p: Patient):
    raw = p.model_dump()
    row = add_features(pd.DataFrame([raw]))[numeric_cols + categorical_cols]
    raw_score = float(pipeline.predict_proba(row)[0][1])

    prob = float(calibrator.predict_proba(logit(np.array([raw_score])))[0][1])
    percent = prob * 100
    if prob < 0.02:
        risk_level = "THẤP"
        advice = "Nguy cơ thấp. Hãy duy trì lối sống lành mạnh."
    elif prob < 0.05:
        risk_level = "TRUNG BÌNH THẤP"
        advice = "Có một số yếu tố nguy cơ. Nên kiểm tra sức khỏe định kỳ."
    elif prob < 0.15:
        risk_level = "TRUNG BÌNH"
        advice = "Nguy cơ đáng chú ý. Nên đi khám bác sĩ và điều chỉnh lối sống."
    elif prob < 0.30:
        risk_level = "CAO"
        advice = "Nguy cơ cao. Nên đi khám chuyên khoa sớm."
    else:
        risk_level = "RẤT CAO"
        advice = "Nguy cơ rất cao. Cần đến bệnh viện kiểm tra càng sớm càng tốt."

    smoking_vi = {"smokes": "đang hút", "formerly smoked": "đã từng hút"}

    factors = []
    if raw["age"] >= 60:
        factors.append(f"Tuổi cao ({raw['age']:.0f} tuổi)")
    if raw["hypertension"] == 1:
        factors.append("Bị cao huyết áp")
    if raw["heart_disease"] == 1:
        factors.append("Có bệnh tim")
    if raw["avg_glucose_level"] >= 140:
        factors.append(f"Đường huyết cao ({raw['avg_glucose_level']})")
    if raw["bmi"] >= 30:
        factors.append(f"Thừa cân / Béo phì (BMI = {raw['bmi']})")
    if raw["smoking_status"] in smoking_vi:
        factors.append(f"Hút thuốc ({smoking_vi[raw['smoking_status']]})")

    warning = ""
    if raw["age"] > max_age_trained:
        warning = f"Tuổi vượt quá dữ liệu huấn luyện (tối đa {max_age_trained}), kết quả chỉ mang tính ước lượng."

    return {
        "probability": round(percent, 2),
        "high_risk": raw_score >= threshold,
        "risk_level": risk_level,
        "advice": advice,
        "factors": factors,
        "warning": warning,
    }