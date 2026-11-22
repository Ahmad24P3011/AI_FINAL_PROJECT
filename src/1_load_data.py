"""Step 1: read the 11 raw season files into one clean, date-sorted table.
Run from the final_project folder:  python src/1_load_data.py
"""
from pathlib import Path

import pandas as pd

RAW, OUT = Path("data/raw"), Path("data/processed")
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]   # date, teams, full-time goals, full-time result


def read_season(path):
    """One season file -> the 6 columns we need, plus a season label taken from the file name."""
    df = pd.read_csv(path, encoding="latin-1")[COLUMNS].dropna()    # dropna removes blank trailing rows
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")        # dates are day/month/year
    df["season"] = path.stem.replace("E0_", "")                     # E0_2015-16.csv -> 2015-16
    return df


def main():
    matches = pd.concat([read_season(p) for p in sorted(RAW.glob("E0_*.csv"))], ignore_index=True)
    matches[["FTHG", "FTAG"]] = matches[["FTHG", "FTAG"]].astype(int)
    duplicates = matches.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()     # the same match listed twice
    matches = matches.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    # Oldest first: every later step walks through time in order, so this order matters.
    matches = matches.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT / "matches.csv", index=False)
    print(f"{duplicates} duplicate matches removed")
    print(f"{len(matches)} matches, {matches['season'].nunique()} seasons, "
          f"{matches['Date'].min().date()} to {matches['Date'].max().date()}")
    print(matches.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
