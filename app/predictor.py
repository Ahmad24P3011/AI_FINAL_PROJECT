"""Prediction layer of the app: loads the SAVED models and each team's saved state, and answers "home team v away team".

Nothing is trained or simulated here, and there is no Streamlit code (so it can be tested alone). It reads files made by src/:
  models/logreg.joblib, random_forest.joblib   the two trained models
  data/processed/team_state.csv                each team's Elo and last-5 goal form after its last 2025-26 match
  data/processed/matches.csv                   used only to show what really happened (never a model input)
"""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIERS = {"logreg": "Logistic Regression (primary)", "random_forest": "Random Forest"}


class AppDataError(Exception):
    """A saved file is missing (the app shows a friendly message)."""


class Predictor:
    def __init__(self):
        needed = [ROOT / "data/processed/team_state.csv", ROOT / "data/processed/matches.csv"] + \
                 [ROOT / f"models/{m}.joblib" for m in CLASSIFIERS]
        for path in needed:
            if not path.exists():
                raise AppDataError(f"app data is missing: {path.name} (run the src/ scripts first)")
        self.state = pd.read_csv(needed[0]).set_index("team")
        self.teams = sorted(self.state.index)
        matches = pd.read_csv(needed[1])
        real = matches[matches["season"] == "2025-26"]                       # every pairing is played once per season
        self.real_results = real.set_index(["HomeTeam", "AwayTeam"])
        self.classifiers = {m: joblib.load(ROOT / f"models/{m}.joblib") for m in CLASSIFIERS}

    def predict(self, home_team, away_team):
        """Everything the page shows for one match, as plain numbers."""
        if home_team == away_team:
            raise ValueError("a team cannot play itself")
        home, away = self.state.loc[home_team], self.state.loc[away_team]
        row = pd.DataFrame([{"home_elo": home["elo"], "away_elo": away["elo"], "elo_diff": home["elo"] - away["elo"],
                             "home_gf_5": home["gf_5"], "home_ga_5": home["ga_5"],
                             "away_gf_5": away["gf_5"], "away_ga_5": away["ga_5"]}])
        probabilities = {m: model.predict_proba(row[model.feature_names_in_])[0] for m, model in self.classifiers.items()}
        return {"home": home_team, "away": away_team, "elo_home": float(home["elo"]), "elo_away": float(away["elo"]),
                "elo_diff": float(home["elo"] - away["elo"]), "probabilities": probabilities,
                "gf_home": float(home["gf_5"]), "gf_away": float(away["gf_5"])}

    def real_result(self, home_team, away_team):
        """The real 2025-26 fixture of this pairing (date, score, outcome), or None. Display only: no model reads it."""
        if (home_team, away_team) not in self.real_results.index:
            return None
        row = self.real_results.loc[(home_team, away_team)]
        home, away = int(row["FTHG"]), int(row["FTAG"])
        return {"date": row["Date"], "home_score": home, "away_score": away, "outcome": 0 if home > away else 1 if home == away else 2}
