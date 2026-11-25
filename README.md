# ML FastAPI Docker

ML-сервис для классификации опухолей (Breast Cancer Wisconsin: malignant / benign).
Модель: StandardScaler + RandomForest (sklearn Pipeline). API на FastAPI, упаковано в Docker.

## Структура
- `train.py` — обучает модель (кросс-валидация, метрики) и сохраняет в `model.joblib`
- `main.py` — FastAPI-приложение (`GET /`, `GET /features`, `POST /predict`)
- `requirements.txt` — зависимости
- `Dockerfile` — инструкции для контейнера

## Запуск локально
```bash
pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload
```
Документация (Swagger): http://127.0.0.1:8000/docs

## Запуск в Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```
Документация: http://localhost:8000/docs

## Эндпоинты
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
