import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_FILE = os.path.join(DATA_DIR, "Training.csv")
TEST_FILE = os.path.join(DATA_DIR, "Testing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "prognosis"
DROP_COLS = ["Unnamed: 133"]

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
for col in DROP_COLS:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

X_train = train.drop(columns=[TARGET]).fillna(0).astype(int)
y_train = train[TARGET].astype(str)
X_test = test.drop(columns=[TARGET]).fillna(0).astype(int)
y_test = test[TARGET].astype(str)

symptoms = X_train.columns.tolist()

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=None),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "Naive Bayes": MultinomialNB(),
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores[name] = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision_macro": round(precision_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
        "recall_macro": round(recall_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
        "f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0) * 100, 2),
    }

# Random Forest is used by the web app because it provides class probabilities
# and feature importance for the educational explanation panel.
final_model = models["Random Forest"]
final_model.fit(X_train, y_train)
joblib.dump(final_model, os.path.join(MODEL_DIR, "random_forest_model.joblib"))
joblib.dump(symptoms, os.path.join(MODEL_DIR, "symptoms.pkl"))

meta = {
    "dataset": "Disease Prediction Using Machine Learning (Kaggle: kaushil268/disease-prediction-using-machine-learning)",
    "train_rows": int(len(train)),
    "unique_training_patterns": int(train.drop_duplicates().shape[0]),
    "test_rows": int(len(test)),
    "symptom_features": int(len(symptoms)),
    "diseases": int(y_train.nunique()),
    "models": scores,
    "final_model": "Random Forest",
    "feature_note": "All 132 symptom columns are binary indicators. Model feature importance is a statistical model signal, not a causal medical explanation.",
    "medical_note": "Educational prototype only. Predictions are not a medical diagnosis and must not be used for treatment decisions."
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))
print("Saved model to models/random_forest_model.joblib")
