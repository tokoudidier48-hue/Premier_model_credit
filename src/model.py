# src/model.py

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


DATA_PATH = "data/UCI_Credit_Card.csv"
MODEL_PATH = "src/model.pkl"
SCALER_PATH = "src/scaler.pkl"


def train_model():
    # Charger les données
    df = pd.read_csv(DATA_PATH)

    # Supprimer la colonne ID si elle existe
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    # Séparer X et y
    X = df.drop("default.payment.next.month", axis=1)
    y = df["default.payment.next.month"]

    # Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Évaluation
    probs = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print("AUC:", auc)

    # Sauvegarde
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model and scaler saved successfully!")


def predict(features):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]

    return probability


if __name__ == "__main__":
    train_model()
