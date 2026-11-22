"""Step 3: train two models on 2015-16 to 2024-25 and save them. The test season (2025-26) is NOT used here.
Run:  python src/3_train.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LR_FEATURES = ["elo_diff", "home_gf_5", "away_gf_5"]
RF_FEATURES = ["home_elo", "away_elo", "elo_diff", "home_gf_5", "home_ga_5", "away_gf_5", "away_ga_5"]
TEST_SEASON = "2025-26"


def main():
    features = pd.read_csv("data/processed/features.csv")
    # Time-based split: whole old seasons for training, the newest season for testing. Never a random split.
    train = features[features["season"] != TEST_SEASON]
    test = features[features["season"] == TEST_SEASON]
    print(f"train: {len(train)} matches ({train['season'].iloc[0]} to {train['season'].iloc[-1]}), "
          f"test: {len(test)} matches ({TEST_SEASON})")

    # Logistic Regression needs features on a similar scale, so the scaler sits inside the pipeline.
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(train[LR_FEATURES], train["result"])

    forest = RandomForestClassifier(random_state=42)       # default settings, fixed seed so it is repeatable
    forest.fit(train[RF_FEATURES], train["result"])

    joblib.dump(logreg, "models/logreg.joblib")
    joblib.dump(forest, "models/random_forest.joblib")
    print("saved models/logreg.joblib      (Logistic Regression on", LR_FEATURES, ")")
    print("saved models/random_forest.joblib (Random Forest on", len(RF_FEATURES), "features)")


if __name__ == "__main__":
    main()
