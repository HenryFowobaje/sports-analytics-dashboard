import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from match_preprocessing import load_and_process_data

def train_model():
    df = load_and_process_data()

    X = df.drop(columns=["match_result"])
    y = df["match_result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🆕 Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/match_predictor.pkl")
    print("✅ Random Forest model saved as models/match_predictor.pkl")

if __name__ == "__main__":
    train_model()
