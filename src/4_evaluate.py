"""Step 4: score both models on the unseen 2025-26 season and compare them with an "always home win" baseline.
Run:  python src/4_evaluate.py
"""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

TEST_SEASON = "2025-26"
LABELS = [0, 1, 2]                                   # 0 = home win, 1 = draw, 2 = away win


def main():
    features = pd.read_csv("data/processed/features.csv")
    test = features[features["season"] == TEST_SEASON]
    y = test["result"]
    rows = []
    for name, file in [("Logistic Regression", "logreg"), ("Random Forest", "random_forest")]:
        model = joblib.load(f"models/{file}.joblib")
        proba = model.predict_proba(test[model.feature_names_in_])      # one probability per outcome
        pred = proba.argmax(axis=1)                                     # the outcome with the highest probability
        matrix = confusion_matrix(y, pred, labels=LABELS)               # rows = what happened, columns = what we predicted
        rows.append({"model": name, "accuracy": accuracy_score(y, pred), "log_loss": log_loss(y, proba, labels=LABELS),
                     "draws_predicted": int((pred == 1).sum()), "confusion_matrix": matrix.tolist()})
        print(f"{name} confusion matrix (rows = actual Home/Draw/Away, columns = predicted):\n{matrix}\n")
    # Baseline: say "home win" every time. It has no probabilities, so there is no log loss for it.
    rows.append({"model": "Always home win", "accuracy": (y == 0).mean(), "log_loss": float("nan"),
                 "draws_predicted": 0, "confusion_matrix": ""})
    table = pd.DataFrame(rows)
    table.to_csv("results/comparison.csv", index=False)
    print(f"Test season {TEST_SEASON}: {len(test)} matches")
    print(table.drop(columns="confusion_matrix").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
