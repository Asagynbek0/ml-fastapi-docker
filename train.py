import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)

# ---------- Setup MLflow ----------
# Все данные экспериментов хранятся в локальной базе mlflow.db
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer-classification")

MODEL_NAME = "breast-cancer-rf"

# ---------- Data ----------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Hyperparametrs ----------
params = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
}

# ---------- Learning by MLflow run ----------
with mlflow.start_run(run_name="random-forest"):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(**params)),
    ])

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Метрики
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "cv_accuracy_mean": cv_scores.mean(),
    }

    # 1. Логируем параметры
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForest + StandardScaler")

    # 2. Логируем метрики
    mlflow.log_metrics(metrics)

    # 3. Сохраняем файл модели для FastAPI и логируем его как артефакт
    artifact = {
        "model": model,
        "feature_names": list(data.feature_names),
        "target_names": [str(n) for n in data.target_names],
    }
    joblib.dump(artifact, "model.joblib")
    mlflow.log_artifact("model.joblib")

    # 4. Логируем модель в формате MLflow и регистрируем в Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        serialization_format="cloudpickle",
        registered_model_name=MODEL_NAME,
    )

    print("=== Метрики ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print(f"Модель зарегистрирована в MLflow Registry как '{MODEL_NAME}'")
    print("Файл model.joblib сохранён для FastAPI")