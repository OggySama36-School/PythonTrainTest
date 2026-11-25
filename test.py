"""Gọi thử service Python (api.py) giống như Go sẽ gọi, và kiểm tra nhanh các ràng buộc.

Điều kiện: service đang chạy (mặc định cổng 8000; đổi bằng biến môi trường API_URL).
Chạy (trong thư mục chứa test.py):  python test.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("API_URL", "http://localhost:8000")

# Dòng đầu tiên của bộ dữ liệu Kaggle (id 9046)
patient = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}

EXPECTED_KEYS = {"probability", "alert_threshold", "high_risk", "risk_level",
                 "advice", "factors", "warning", "disclaimer"}


def call(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, {"raw": body}


failures = 0


def check(name, condition, detail=""):
    global failures
    print(f"  [{'OK' if condition else 'LỖI'}] {name}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures += 1


try:
    print("1) Bệnh nhân mẫu")
    status, r = call("POST", "/predict", patient)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    check("HTTP 200", status == 200, status)
    check("đủ các khóa trả về", EXPECTED_KEYS <= set(r), sorted(EXPECTED_KEYS - set(r)))
    check("high_risk khớp với risk_level (CAO / RẤT CAO)",
          r.get("high_risk") == (r.get("risk_level") in ("CAO", "RẤT CAO")))

    print("\n2) Bỏ trống BMI (phải dùng giá trị trung vị, kèm cảnh báo)")
    status, r = call("POST", "/predict", {**patient, "bmi": None})
    check("HTTP 200", status == 200, status)
    check("có cảnh báo về BMI", "BMI" in r.get("warning", ""), r.get("warning"))

    print("\n3) Tuổi nằm ngoài khoảng dữ liệu huấn luyện")
    status, r = call("POST", "/predict", {**patient, "age": 100})
    check("HTTP 200", status == 200, status)
    check("có cảnh báo", bool(r.get("warning")))

    print("\n4) Dữ liệu sai phải bị từ chối (HTTP 422)")
    for label, bad in [("tuổi âm", {**patient, "age": -5}),
                       ("work_type không hợp lệ", {**patient, "work_type": "Student"}),
                       ("BMI phi thực tế", {**patient, "bmi": 500}),
                       ("thiếu trường gender", {k: v for k, v in patient.items() if k != "gender"})]:
        status, _ = call("POST", "/predict", bad)
        check(label, status == 422, f"nhận HTTP {status}")

    print("\n5) /model-info")
    status, info = call("GET", "/model-info")
    check("HTTP 200", status == 200, status)
    check("có tên mô hình và ngưỡng", {"model_name", "threshold"} <= set(info))
    if status == 200:
        print(f"  Mô hình: {info.get('model_name')} | ngưỡng cảnh báo: {info.get('threshold')}")

except urllib.error.URLError:
    print(f"Không kết nối được tới {BASE}. Service Python chưa chạy?")
    sys.exit(1)

print("\n" + ("TẤT CẢ ĐỀU ĐẠT" if failures == 0 else f"CÓ {failures} MỤC LỖI"))
sys.exit(1 if failures else 0)