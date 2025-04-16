import pandas as pd
import joblib
import matplotlib.pyplot as plt

def plot_feature_importance(model_path="models/match_predictor.pkl", feature_names=None):
    model = joblib.load(model_path)

    if not hasattr(model, "feature_importances_"):
        print("❌ This model does not support feature importances.")
        return

    importances = model.feature_importances_

    # Use dummy feature names if not provided
    if feature_names is None:
        feature_names = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("🔍 Top Features:\n", df.head())

    # Optional: visualize
    df.plot(kind="barh", x="Feature", y="Importance", legend=False, figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_feature_importance()
