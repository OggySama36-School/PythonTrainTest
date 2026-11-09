"""Gọi thử service Python (api.py) giống như Go sẽ gọi.

Điều kiện: service đang chạy ở cổng 8000.
Chạy (trong thư mục train/):  .venv\\Scripts\\python.exe test.py
"""

import json
import urllib.error
import urllib.request

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

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(patient).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req) as res:
        result = json.loads(res.read())
        print(json.dumps(result, indent=2, ensure_ascii=False))
except urllib.error.HTTPError as e:
    print("Lỗi", e.code, ":", e.read().decode("utf-8"))
except urllib.error.URLError:
    print("Không kết nối được. Service Python chưa chạy ở cổng 8000")