import os

import requests
import streamlit as st

# Адрес FastAPI (можно переопределить переменной окружения API_URL)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Два реальных примера из датасета Breast Cancer Wisconsin
MALIGNANT = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]
BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259,
]

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Classifier")
st.write(
    "Введите 30 признаков опухоли (или выберите пример) и нажмите **Predict**. "
    "Запрос уходит в FastAPI, который использует обученную модель."
)

# Получаем названия признаков из API
try:
    resp = requests.get(f"{API_URL}/features", timeout=5)
    resp.raise_for_status()
    feature_names = resp.json()["feature_names"]
except Exception as e:
    st.error(f"Не удалось подключиться к API ({API_URL}). Запущен ли сервер? Ошибка: {e}")
    st.stop()

# Выбор примера
st.subheader("1. Выберите пример данных")
example = st.radio(
    "Пример",
    ["Злокачественная (malignant)", "Доброкачественная (benign)"],
    horizontal=True,
)
defaults = MALIGNANT if example.startswith("Злокач") else BENIGN

# Поля ввода признаков
st.subheader("2. При желании измените значения")
values = []
cols = st.columns(3)
for i, name in enumerate(feature_names):
    with cols[i % 3]:
        values.append(
            st.number_input(
                name,
                value=float(defaults[i]),
                format="%.5f",
                key=f"{example}-{name}",  # при смене примера поля обновляются
            )
        )

# Prediction
st.subheader("3. Результат")
if st.button("Predict", type="primary"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"features": values}, timeout=10)
        r.raise_for_status()
        result = r.json()
        if result["class_name"] == "malignant":
            st.error(f"Предсказание: **{result['class_name']}** (злокачественная)")
        else:
            st.success(f"Предсказание: **{result['class_name']}** (доброкачественная)")
        st.write("Вероятности:")
        st.bar_chart(result["probabilities"])
    except Exception as e:
        st.error(f"Ошибка запроса к API: {e}")