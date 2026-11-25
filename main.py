import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ML API",
    description="Breast Cancer Wisconsin: классификация опухоли (malignant / benign)",
)

# Загружаем модель один раз при старте
artifact = joblib.load("model.joblib")
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
TARGET_NAMES = artifact["target_names"]

# Пример: реальная запись из датасета (злокачественная опухоль)
EXAMPLE = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 числовых признаков опухоли в порядке из GET /features",
    )

    model_config = {"json_schema_extra": {"examples": [{"features": EXAMPLE}]}}


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/features")
def features():
    """Список из 30 признаков в нужном порядке."""
    return {"count": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        data = [request.features]
        pred = int(model.predict(data)[0])
        proba = model.predict_proba(data)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "prediction": pred,
        "class_name": TARGET_NAMES[pred],
        "probabilities": {
            TARGET_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }
