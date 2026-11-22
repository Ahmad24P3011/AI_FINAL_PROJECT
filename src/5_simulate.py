"""Step 5: play the 2025-26 season 100 times. Each match's result is sampled from the Logistic Regression probabilities.
Run:  python src/5_simulate.py
NOTE: the probabilities use each fixture's saved pre-match features. For 2025-26 those were built from real earlier
results, so this is a replay with random outcomes, NOT a true pre-season forecast.
"""
import joblib
import numpy as np
import pandas as pd

N_SIMS = 100
HOME_PTS, AWAY_PTS = np.array([3, 1, 0]), np.array([0, 1, 3])     # points for a home win / draw / away win


def main():
    model = joblib.load("models/logreg.joblib")
    fixtures = pd.read_csv("data/processed/features.csv")
    fixtures = fixtures[fixtures["season"] == "2025-26"].reset_index(drop=True)
    proba = model.predict_proba(fixtures[model.feature_names_in_])             # (380 matches, 3 outcomes)
    teams = sorted(set(fixtures["HomeTeam"]))
    home, away = (fixtures[c].map({t: i for i, t in enumerate(teams)}).to_numpy() for c in ("HomeTeam", "AwayTeam"))

    rng = np.random.default_rng(42)                                            # fixed seed: same answer every run
    # Sample an outcome per match and simulation: compare one random number with the cumulative probabilities.
    outcome = (rng.random((N_SIMS, len(fixtures)))[..., None] > proba.cumsum(axis=1)[:, :2]).sum(axis=2)
    points = np.zeros((N_SIMS, len(fixtures), len(teams)))                     # points won by each team in each match
    match = np.arange(len(fixtures))
    points[:, match, home], points[:, match, away] = HOME_PTS[outcome], AWAY_PTS[outcome]
    total = points.sum(axis=1)                                                 # (simulations, teams) final points
    # Position 1 = champion; a tiny random number breaks ties between teams level on points.
    position = (-(total + rng.random(total.shape) * 0.5)).argsort(axis=1).argsort(axis=1) + 1

    table = pd.DataFrame({"team": teams, "mean_points": total.mean(axis=0), "mean_position": position.mean(axis=0),
                          "P_title": (position == 1).mean(axis=0), "P_top4": (position <= 4).mean(axis=0),
                          "P_relegation": (position >= 18).mean(axis=0)}).sort_values("mean_points", ascending=False)
    table.to_csv("results/simulated_table.csv", index=False)
    # Two extra files only the app's charts need: position probabilities and points after each game.
    pd.DataFrame([(position == p).mean(axis=0) for p in range(1, 21)], index=range(1, 21), columns=teams).T.rename_axis("team").to_csv("results/position_probs.csv")
    by_game = []
    for t, team in enumerate(teams):
        played = np.flatnonzero((home == t) | (away == t))                     # this team's 38 fixtures, in date order
        q = np.percentile(points[:, played, t].cumsum(axis=1), [5, 25, 50, 75, 95], axis=0)
        by_game += [{"team": team, "game_no": g + 1, **dict(zip(["p5", "p25", "p50", "p75", "p95"], q[:, g]))} for g in range(len(played))]
    pd.DataFrame(by_game).to_csv("results/points_by_game.csv", index=False)

    print(f"{N_SIMS} simulated seasons. Top 6 by mean points:")
    print(table.head(6).round(3).to_string(index=False))
    print("\nBottom 3:")
    print(table.tail(3).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
