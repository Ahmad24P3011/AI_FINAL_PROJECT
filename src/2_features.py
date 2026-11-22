"""Step 2: one row per match, built ONLY from matches played before it (Elo rating + last-5 goal form).
Run:  python src/2_features.py
"""
from collections import defaultdict, deque

import pandas as pd

K, H, START, KEEP, WINDOW = 30, 50, 1500, 0.9, 5     # Elo speed, home bonus, start rating, season carry-over, form window
DEFAULT_GOALS = 1.4                                  # about the league average; used when a team has no history yet
SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}               # Elo "actual score" for the home team
RESULT = {"H": 0, "D": 1, "A": 2}                    # target: 0 = home win, 1 = draw, 2 = away win


def form(games):
    """Average goals scored and conceded over a team's last matches."""
    if not games:
        return DEFAULT_GOALS, DEFAULT_GOALS
    return sum(g[0] for g in games) / len(games), sum(g[1] for g in games) / len(games)


def main():
    matches = pd.read_csv("data/processed/matches.csv", parse_dates=["Date"])
    elo = defaultdict(lambda: START)
    recent = defaultdict(lambda: deque(maxlen=WINDOW))     # (goals for, goals against), kept across seasons
    rows, season = [], None
    for m in matches.itertuples():
        if m.season != season:                             # new season: pull every rating 10% back toward 1500
            for team in elo:
                elo[team] = START + KEEP * (elo[team] - START)
            season = m.season
        h, a = m.HomeTeam, m.AwayTeam
        (hgf, hga), (agf, aga) = form(recent[h]), form(recent[a])
        # 1) features come from the past only ...
        rows.append({"Date": m.Date, "season": m.season, "HomeTeam": h, "AwayTeam": a,
                     "home_elo": elo[h], "away_elo": elo[a], "elo_diff": elo[h] - elo[a],
                     "home_gf_5": hgf, "home_ga_5": hga, "away_gf_5": agf, "away_ga_5": aga, "result": RESULT[m.FTR]})
        # 2) ... and only now do we learn this match's result
        expected = 1 / (1 + 10 ** (-(elo[h] + H - elo[a]) / 400))
        change = K * (SCORE[m.FTR] - expected)             # the winner gains what the loser loses
        elo[h] += change
        elo[a] -= change
        recent[h].append((m.FTHG, m.FTAG))
        recent[a].append((m.FTAG, m.FTHG))
    features = pd.DataFrame(rows)
    features.to_csv("data/processed/features.csv", index=False)
    # Each 2025-26 team's rating and form after its last match: the app uses this for hypothetical match-ups.
    now = matches[matches["season"] == matches["season"].iloc[-1]]
    state = [{"team": t, "elo": elo[t], "gf_5": form(recent[t])[0], "ga_5": form(recent[t])[1]}
             for t in sorted(set(now["HomeTeam"]))]
    pd.DataFrame(state).to_csv("data/processed/team_state.csv", index=False)
    print("feature table shape:", features.shape)
    print(features.drop(columns=["Date", "season", "HomeTeam", "AwayTeam"]).describe().loc[["mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
