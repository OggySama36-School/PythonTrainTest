# -*- coding: utf-8 -*-
"""Huấn luyện mô hình dự đoán nguy cơ đột quỵ (Logistic Regression + Random Forest + Gradient Boosting).

Quy trình:
    1. Tách 20% dữ liệu làm tập TEST, chỉ dùng để chấm điểm cuối cùng.
    2. 80% còn lại chạy cross-validation 5 lần: mọi mẫu đều được dự đoán khi mô hình chưa nhìn thấy nó.
    3. Từ các dự đoán đó: chọn ngưỡng cảnh báo và học phép hiệu chỉnh xác suất.
    4. Huấn luyện mô hình cuối trên 80%, chấm trên TEST, lưu model.

Chạy từ thư mục gốc backend:  python train.py
Kết quả:
    - train/model.joblib  (api.py đọc file này)
    - train/charts/*.png  (biểu đồ cho báo cáo)
"""

# ====================== 1. IMPORT ======================
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, brier_score_loss, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             precision_recall_curve, classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ====================== 2. ĐƯỜNG DẪN & ĐỌC DỮ LIỆU ======================
BASE_DIR = Path(__file__).resolve().parent          # thư mục train/
CSV_PATH = BASE_DIR.parent / "dataset" / "healthcare-dataset-stroke-data.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
MAX_AGE_TRAINED = int(df['age'].max())

print("Shape:", df.shape)
print(df['stroke'].value_counts())

# ====================== 3. TẠO ĐẶC TRƯNG ======================
# LƯU Ý: hàm này phải GIỐNG HỆT hàm add_features trong api.py
# bmi thiếu được điền median bên trong pipeline (bước 4), không điền trước khi chia dữ liệu
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


df = add_features(df)

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                'age_group', 'is_senior', 'bmi_obese', 'glucose_high',
                'comorbidity', 'smoking_binary', 'risk_score']

X = df[numeric_cols + categorical_cols]
y = df['stroke']

# Tách tập TEST (20%). Phần còn lại (dev) dùng cho cross-validation
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nDev : {len(y_dev)} mẫu, {int(y_dev.sum())} ca đột quỵ")
print(f"Test: {len(y_test)} mẫu, {int(y_test.sum())} ca đột quỵ")

# ====================== 4. ĐỊNH NGHĨA MÔ HÌNH ======================
def make_pipeline(model):
    # Cột số: điền thiếu bằng median rồi chuẩn hóa. Cột chữ: one-hot
    prep = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ])
    return Pipeline([('prep', prep), ('clf', model)])


def make_lr():
    return LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)

def make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                  class_weight='balanced', random_state=42, n_jobs=-1)

def make_gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                      random_state=42)

def make_ensemble():
    # Gộp 3 mô hình bằng cách lấy trung bình xác suất
    return make_pipeline(VotingClassifier(
        estimators=[('lr', make_lr()), ('rf', make_rf()), ('gb', make_gb())],
        voting='soft'))

def make_baseline():
    # Mô hình nền chỉ dùng tuổi, để chứng minh các yếu tố khác có đóng góp
    return Pipeline([
        ('sel', ColumnTransformer([('age', StandardScaler(), ['age'])])),
        ('clf', make_lr()),
    ])


models = {
    'Chỉ dùng tuổi (mô hình nền)': make_baseline,
    'Logistic Regression': lambda: make_pipeline(make_lr()),
    'Random Forest': lambda: make_pipeline(make_rf()),
    'Gradient Boosting': lambda: make_pipeline(make_gb()),
    'Ensemble (LR + RF + GB)': make_ensemble,
}

# ====================== 5. CROSS-VALIDATION 5 LẦN ======================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def cv_oof(make_model):
    """Trả về dự đoán 'ngoài mẫu' cho toàn bộ tập dev và AUC của từng fold."""
    oof = np.zeros(len(y_dev))
    aucs = []
    for tr, va in skf.split(X_dev, y_dev):
        m = make_model().fit(X_dev.iloc[tr], y_dev.iloc[tr])
        oof[va] = m.predict_proba(X_dev.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y_dev.iloc[va], oof[va]))
    return oof, np.array(aucs)


print("\nĐang chạy cross-validation 5 lần (mất vài phút)...")
oof_probs, cv_aucs = {}, {}
for name, make_model in models.items():
    oof_probs[name], cv_aucs[name] = cv_oof(make_model)
    print(f"  xong: {name}")

ENS = 'Ensemble (LR + RF + GB)'

# ====================== 6. CHỌN NGƯỠNG & HIỆU CHỈNH TỪ DỰ ĐOÁN NGOÀI MẪU ======================
oof = oof_probs[ENS]
precisions, recalls, thresholds = precision_recall_curve(y_dev, oof)

# precisions/recalls dài hơn thresholds 1 phần tử, nên cắt [:-1] để khớp
p = precisions[:-1]
r = recalls[:-1]
f1_scores = 2 * p * r / (p + r + 1e-8)
threshold_f1 = float(thresholds[np.argmax(f1_scores)])

# Ngưỡng "y tế": ngưỡng cao nhất mà vẫn bắt được ít nhất 85% ca đột quỵ (recall >= 0.85)
ok = r >= 0.85
THRESHOLD_RAW = float(thresholds[ok].max()) if ok.any() else threshold_f1

# Hiệu chỉnh (Platt scaling): học phép ánh xạ từ điểm thô sang tỉ lệ đột quỵ thật.
# Cần vì class_weight='balanced' làm điểm thô cao hơn tỉ lệ thật.
def logit(prob):
    prob = np.clip(prob, 1e-4, 1 - 1e-4)
    return np.log(prob / (1 - prob)).reshape(-1, 1)


calibrator = LogisticRegression(C=1e6).fit(logit(oof), y_dev)
THRESHOLD_CAL = float(calibrator.predict_proba(logit(np.array([THRESHOLD_RAW])))[0][1])

print(f"\nNgưỡng cảnh báo (điểm thô)       : {THRESHOLD_RAW:.3f}")
print(f"Ngưỡng cảnh báo (sau hiệu chỉnh) : {THRESHOLD_CAL * 100:.1f}%")

# ====================== 7. MÔ HÌNH CUỐI & CHẤM TRÊN TẬP TEST ======================
print("\nĐang huấn luyện mô hình cuối trên toàn bộ tập dev...")
fitted = {name: make_model().fit(X_dev, y_dev) for name, make_model in models.items()}
final_pipeline = fitted[ENS]

test_probs = {name: m.predict_proba(X_test)[:, 1] for name, m in fitted.items()}
raw_test = test_probs[ENS]
cal_test = calibrator.predict_proba(logit(raw_test))[:, 1]

print("\n" + "=" * 78)
print("SO SÁNH CÁC MÔ HÌNH: AUC")
print("=" * 78)
cmp_df = pd.DataFrame({
    'Mô hình': list(models.keys()),
    'AUC cross-validation (TB ± độ lệch)': [f"{cv_aucs[n].mean():.3f} ± {cv_aucs[n].std():.3f}" for n in models],
    'AUC trên tập test': [round(roc_auc_score(y_test, test_probs[n]), 3) for n in models],
})
print(cmp_df.to_string(index=False))


def wilson(k, n, z=1.96):
    """Khoảng tin cậy 95% cho một tỉ lệ k/n."""
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


y_pred = (raw_test >= THRESHOLD_RAW).astype(int)
tp = int(((y_pred == 1) & (y_test == 1)).sum())
lo, hi = wilson(tp, int(y_test.sum()))

print("\n" + "=" * 78)
print(f"ENSEMBLE TRÊN TẬP TEST (ngưỡng cảnh báo = {THRESHOLD_RAW:.3f})")
print("=" * 78)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}  ({tp}/{int(y_test.sum())} ca; khoảng tin cậy 95%: {lo:.2f} - {hi:.2f})")
print(f"  F1-score  : {f1_score(y_test, y_pred):.4f}")
print(f"  AUC-ROC   : {roc_auc_score(y_test, raw_test):.4f}")
print(classification_report(y_test, y_pred, target_names=['Không đột quỵ', 'Có đột quỵ']))

base_rate_dev = float(y_dev.mean())
print("Chất lượng xác suất trên tập test:")
print(f"  Brier score chưa hiệu chỉnh : {brier_score_loss(y_test, raw_test):.4f}")
print(f"  Brier score sau hiệu chỉnh  : {brier_score_loss(y_test, cal_test):.4f}")
print(f"  Brier của mức nền (luôn đoán {base_rate_dev * 100:.1f}%): {brier_score_loss(y_test, np.full(len(y_test), base_rate_dev)):.4f}")
print(f"  Tỉ lệ đột quỵ thật: {y_test.mean() * 100:.1f}% | điểm TB chưa hiệu chỉnh: {raw_test.mean() * 100:.1f}% | sau hiệu chỉnh: {cal_test.mean() * 100:.1f}%")

# Bảng độ tin cậy: chia test thành 5 nhóm theo điểm, so điểm dự đoán với tỉ lệ thật
rel = pd.DataFrame({'cal': cal_test, 'raw': raw_test, 'y': y_test.values})
rel['nhom'] = pd.qcut(rel['cal'].rank(method='first'), 5, labels=False)
rel_g = rel.groupby('nhom').agg(n=('y', 'size'), raw=('raw', 'mean'), cal=('cal', 'mean'), real=('y', 'mean'))
print("\nĐộ tin cậy theo 5 nhóm (từ nguy cơ thấp đến cao):")
print(f"  {'Nhóm':<6}{'Số người':>10}{'Điểm thô':>12}{'Sau hiệu chỉnh':>18}{'Tỉ lệ thật':>14}")
for i, row in rel_g.iterrows():
    print(f"  {i + 1:<6}{int(row['n']):>10}{row['raw'] * 100:>11.1f}%{row['cal'] * 100:>17.1f}%{row['real'] * 100:>13.1f}%")

# ====================== 8. VẼ BIỂU ĐỒ ======================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 8.1 Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Không ĐQ', 'Có ĐQ']).plot(
    cmap='Blues', ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title(f'Ma trận nhầm lẫn (ngưỡng = {THRESHOLD_RAW:.2f})')

# 8.2 ROC trên tập test
for name, pr in test_probs.items():
    RocCurveDisplay.from_predictions(y_test, pr, name=name, ax=axes[0, 1])
axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0, 1].set_title('So sánh ROC (tập test)')
axes[0, 1].grid(alpha=0.3)

# 8.3 AUC cross-validation
names = list(models.keys())
axes[0, 2].barh(names, [cv_aucs[n].mean() for n in names],
                xerr=[cv_aucs[n].std() for n in names], color='steelblue')
axes[0, 2].set_xlim(0.5, 1.0)
axes[0, 2].set_xlabel('AUC (trung bình ± độ lệch chuẩn, 5 fold)')
axes[0, 2].set_title('AUC cross-validation')

# 8.4 Precision - Recall trên dự đoán ngoài mẫu
axes[1, 0].plot(recalls, precisions, color='#1f77b4', lw=2)
axes[1, 0].axvline(0.85, color='green', ls='--', alpha=0.5, label='Recall = 85%')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Đường Precision-Recall (cross-validation)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 8.5 Độ tin cậy xác suất
axes[1, 1].plot([0, 0.3], [0, 0.3], 'k--', alpha=0.3, label='Lý tưởng')
axes[1, 1].plot(rel_g['raw'], rel_g['real'], 'o-', color='indianred', label='Chưa hiệu chỉnh')
axes[1, 1].plot(rel_g['cal'], rel_g['real'], 'o-', color='seagreen', label='Sau hiệu chỉnh')
axes[1, 1].set_xlabel('Xác suất dự đoán trung bình của nhóm')
axes[1, 1].set_ylabel('Tỉ lệ đột quỵ thật của nhóm')
axes[1, 1].set_title('Độ tin cậy của xác suất (tập test)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 8.6 Hệ số của Logistic Regression
lr_pipe = fitted['Logistic Regression']
feature_names = numeric_cols + list(
    lr_pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
)
coefs = lr_pipe.named_steps['clf'].coef_[0]
imp = pd.DataFrame({'Feature': feature_names, 'Coef': coefs, 'Abs': np.abs(coefs)})
imp = imp.sort_values('Abs', ascending=True).tail(12)
axes[1, 2].barh(imp['Feature'], imp['Coef'], color=np.where(imp['Coef'] > 0, 'indianred', 'steelblue'))
axes[1, 2].set_xlabel('Hệ số (đỏ: tăng nguy cơ, xanh: giảm nguy cơ)')
axes[1, 2].set_title('Top 12 yếu tố ảnh hưởng (Logistic Regression)')

plt.tight_layout()
plt.savefig(CHART_DIR / "evaluation_plots.png", dpi=120, bbox_inches='tight')
plt.show()

# ====================== 9. LƯU MODEL ======================
# Gói tất cả những gì api.py cần vào một file
joblib.dump({
    'pipeline': final_pipeline,
    'calibrator': calibrator,
    'threshold': THRESHOLD_RAW,          # ngưỡng trên điểm thô
    'threshold_calibrated': THRESHOLD_CAL,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'max_age_trained': MAX_AGE_TRAINED,
}, MODEL_PATH)

print(f"\nĐã lưu model vào: {MODEL_PATH}")

# ====================== 10. KIỂM TRA NHANH ======================
cases = [
    ("90 tuổi, đủ yếu tố nguy cơ",
     dict(gender='Male', age=90, hypertension=1, heart_disease=1, ever_married='Yes',
          work_type='Private', Residence_type='Urban', avg_glucose_level=250, bmi=35,
          smoking_status='smokes')),
    ("70 tuổi, cao huyết áp + bệnh tim",
     dict(gender='Female', age=70, hypertension=1, heart_disease=1, ever_married='Yes',
          work_type='Private', Residence_type='Rural', avg_glucose_level=180, bmi=32,
          smoking_status='formerly smoked')),
    ("55 tuổi, cao huyết áp nhẹ",
     dict(gender='Male', age=55, hypertension=1, heart_disease=0, ever_married='Yes',
          work_type='Private', Residence_type='Urban', avg_glucose_level=110, bmi=28,
          smoking_status='formerly smoked')),
    ("25 tuổi, khỏe mạnh",
     dict(gender='Female', age=25, hypertension=0, heart_disease=0, ever_married='No',
          work_type='Private', Residence_type='Urban', avg_glucose_level=85, bmi=22,
          smoking_status='never smoked')),
]

print("\n----- KIỂM TRA NHANH (điểm tăng dần theo mức nguy cơ là đúng) -----")
for title, case in cases:
    row = add_features(pd.DataFrame([case]))[numeric_cols + categorical_cols]
    raw = final_pipeline.predict_proba(row)[0][1]
    cal = calibrator.predict_proba(logit(np.array([raw])))[0][1]
    flag = "CẢNH BÁO" if raw >= THRESHOLD_RAW else "bình thường"
    print(f"  {title:<36} {cal * 100:5.1f}%   {flag}")