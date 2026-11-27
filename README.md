# ML FastAPI Docker + MLflow + Streamlit

ML-система для классификации опухолей (Breast Cancer Wisconsin: malignant / benign).
Модель: StandardScaler + RandomForest (sklearn Pipeline).

Проект состоит из четырёх частей:
- **Обучение + MLflow** (`train.py`): трекинг экспериментов и Model Registry
- **API** (`main.py`): FastAPI, отдаёт предсказания
- **Frontend** (`frontend.py`): страница на Streamlit
- **Docker** (`Dockerfile`): контейнер с API

## Структура
- `train.py` — обучает модель, логирует параметры, метрики и артефакты в MLflow, регистрирует модель в Model Registry, сохраняет `model.joblib`
- `main.py` — FastAPI-приложение (`GET /`, `GET /features`, `POST /predict`)
- `frontend.py` — интерфейс на Streamlit
- `requirements.txt` — зависимости API и обучения
- `requirements-frontend.txt` — зависимости frontend
- `Dockerfile` — инструкции для контейнера

## Установка
```bash
pip install -r requirements.txt
pip install -r requirements-frontend.txt
```

## 1. Обучение и MLflow
```bash
python train.py
```
Что происходит:
- создаётся эксперимент `breast-cancer-classification`
- логируются параметры (`n_estimators`, `max_depth`, `random_state`)
- логируются метрики (accuracy, F1-score, precision, recall)
- файл `model.joblib` сохраняется как артефакт
- модель регистрируется в Model Registry под именем `breast-cancer-rf` (версия присваивается автоматически)

Интерфейс MLflow:
```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Открыть: http://127.0.0.1:5000

## 2. Запуск API
```bash
python -m uvicorn main:app --reload
```
Документация (Swagger): http://127.0.0.1:8000/docs

## 3. Запуск frontend (Streamlit)
В отдельном терминале (API должен быть запущен):
```bash
python -m streamlit run frontend.py
```
Открыть: http://localhost:8501

## Запуск API в Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```
Документация: http://localhost:8000/docs

## Эндпоинты API
- `GET /` — проверка, что API работает
- `GET /features` — список из 30 признаков
- `POST /predict` — принимает 30 признаков, возвращает диагноз и вероятности

Пример тела запроса `POST /predict`:
```json
{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
              1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
              25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}
```
Ответ: `{"prediction": 0, "class_name": "malignant", "probabilities": {...}}`