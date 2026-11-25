# -*- coding: utf-8 -*-
"""Service dự đoán nguy cơ đột quỵ: nhận JSON từ Go (Gin), trả JSON về cho Go.

Chạy (trong thư mục chứa api.py):  python -m uvicorn api:app --port 8000
Thử nhanh không cần Go:            mở http://localhost:8000/docs

Model được đọc từ outputs/stroke_model.pkl (do train.py tạo ra).
Có thể đổi đường dẫn bằng biến môi trường STROKE_MODEL_PATH.
"""
import os
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("STROKE_MODEL_PATH", BASE_DIR / "model.joblib"))

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]            # chỉ gồm tiền xử lý + mô hình (SMOTE không cần khi dự đoán)
FEATURES = bundle["features"]
threshold = bundle["threshold"]          # ngưỡng cảnh báo (chọn từ CV theo mục tiêu Recall)
risk_bands = bundle["risk_bands"]        # các mức nguy cơ chia quanh ngưỡng cảnh báo
info = bundle["info"]
ranges = info["numeric_ranges"]          # khoảng giá trị trong dữ liệu huấn luyện

if info.get("sklearn_version") != sklearn.__version__:
    print(f"[CẢNH BÁO] Model huấn luyện bằng scikit-learn {info.get('sklearn_version')}, "
          f"đang chạy {sklearn.__version__}. Hãy cài đúng phiên bản (xem requirements.txt).")

DISCLAIMER = ("Điểm này do mô hình học máy tính từ dữ liệu mẫu, chỉ mang tính minh họa học thuật, "
              "KHÔNG phải xác suất đột quỵ thực tế và không thay thế chẩn đoán của bác sĩ.")

ADVICE = {
    "THẤP": "Điểm nguy cơ thấp. Hãy duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ.",
    "TRUNG BÌNH": "Có một số yếu tố nguy cơ. Nên theo dõi huyết áp, đường huyết và kiểm tra sức khỏe định kỳ.",
    "CAO": "Điểm nguy cơ vượt ngưỡng cảnh báo của mô hình. Nên đặt lịch khám để được bác sĩ đánh giá.",
    "RẤT CAO": "Điểm nguy cơ rất cao so với ngưỡng cảnh báo. Nên đi khám chuyên khoa sớm.",
}

app = FastAPI(title="Stroke risk service")


# Health check: dùng cho Render và UptimeRobot (GET + HEAD) để giữ service luôn thức
@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Thông tin mô hình cho frontend/báo cáo (tên mô hình, ngưỡng, chỉ số trên tập Test...)."""
    return info


class Patient(BaseModel):
    gender: Literal['Male', 'Female', 'Other']
    age: float = Field(gt=0, le=120)
    hypertension: Literal[0, 1]              # tiền sử tăng huyết áp (có/không)
    heart_disease: Literal[0, 1]
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float = Field(ge=20, le=700)
    bmi: Optional[float] = Field(default=None, ge=8, le=100)   # để trống -> dùng giá trị trung vị
    smoking_status: Literal['never smoked', 'formerly smoked', 'smokes', 'Unknown']


def level_of(score: float) -> str:
    for band in risk_bands:
        if score < band["upper"]:
            return band["level"]
    return risk_bands[-1]["level"]


def out_of_range_warnings(raw: dict) -> list:
    msgs = []
    labels = {"age": "Tuổi", "avg_glucose_level": "Đường huyết", "bmi": "BMI"}
    for col, label in labels.items():
        value = raw.get(col)
        if value is None:
            continue
        lo, hi = ranges[col]
        if value < lo or value > hi:
            msgs.append(f"{label} nằm ngoài khoảng dữ liệu huấn luyện ({lo:g} - {hi:g}), "
                        "kết quả chỉ mang tính ước lượng.")
    if raw.get("bmi") is None:
        msgs.append("Chưa nhập BMI, hệ thống dùng giá trị trung vị của dữ liệu huấn luyện.")
    return msgs


@app.post("/predict")
def predict(p: Patient):
    raw = p.model_dump()
    warnings_text = " ".join(out_of_range_warnings(raw))   # tính trước khi thay bmi=None bằng NaN
    row_data = dict(raw)
    if row_data["bmi"] is None:
        row_data["bmi"] = np.nan             # imputer (median) trong pipeline sẽ điền
    row = pd.DataFrame([row_data])[FEATURES]
    score = float(pipeline.predict_proba(row)[0][1])
    level = level_of(score)

    smoking_vi = {"smokes": "đang hút", "formerly smoked": "đã từng hút"}
    factors = []                             # yếu tố nguy cơ đã biết (luật cố định, không phải giải thích của mô hình)
    if raw["age"] >= 60:
        factors.append(f"Tuổi cao ({raw['age']:.0f} tuổi)")
    if raw["hypertension"] == 1:
        factors.append("Có tiền sử tăng huyết áp")
    if raw["heart_disease"] == 1:
        factors.append("Có bệnh tim")
    if raw["avg_glucose_level"] >= 140:
        factors.append(f"Đường huyết cao ({raw['avg_glucose_level']})")
    if raw["bmi"] is not None and raw["bmi"] >= 30:
        factors.append(f"Thừa cân / Béo phì (BMI = {raw['bmi']})")
    if raw["smoking_status"] in smoking_vi:
        factors.append(f"Hút thuốc ({smoking_vi[raw['smoking_status']]})")

    return {
        "probability": round(score * 100, 2),            # giữ tên khóa cũ; ý nghĩa: ĐIỂM nguy cơ (%)
        "alert_threshold": round(threshold * 100, 2),    # ngưỡng cảnh báo (%), cùng thang với probability
        "high_risk": score >= threshold,                 # luôn khớp với mức CAO / RẤT CAO
        "risk_level": level,
        "advice": ADVICE[level],
        "factors": factors,
        "warning": warnings_text,
        "disclaimer": DISCLAIMER,
    }