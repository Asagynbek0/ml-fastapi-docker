import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Загружаем датасет (569 пациентов, 30 признаков)
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Размер данных: {X.shape}")
print(f"Классы: {list(data.target_names)}")

# 2. Делим на train/test (с сохранением пропорции классов)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline: масштабирование + Random Forest
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])

# 4. Кросс-валидация (5 фолдов) на обучающей выборке
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 5. Обучение и оценка на тесте
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Сохраняем модель вместе с метаданными
artifact = {
    "model": model,
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
}
joblib.dump(artifact, "model.joblib")
print("Модель сохранена в model.joblib")
