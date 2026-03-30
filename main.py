from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.train import get_models
from src.evaluate import evaluate

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# 1. Загружаем данные
data = load_data("data/russian_credit_data.csv")

# 2. Делим на признаки и target
y = data["target"]
X = data.drop(columns=["target"])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4. Препроцессор и модели
preprocessor = build_preprocessor(X)
models = get_models()

results = []
fitted_pipelines = {}

# 5. Обучение и оценка
for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba)

    print(name, metrics)

    results.append([name, metrics["roc_auc"], metrics["f1"]])
    fitted_pipelines[name] = pipe

# 6. Сохраняем таблицу результатов
results_df = pd.DataFrame(results, columns=["model", "roc_auc", "f1"])
results_df.to_csv("outputs/results.csv", index=False)

# 7. ROC-кривые
plt.figure(figsize=(8, 6))

for name, pipe in fitted_pipelines.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc.png")
plt.show()

print("\nГотово.")
print("Сохранены:")
print("outputs/results.csv")
print("outputs/roc.png")