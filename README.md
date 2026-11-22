# Premier League Match Prediction 

Predicts Premier League match results (home win / draw / away win) from team Elo ratings and recent goals, compares two models on the unseen 2025-26 season, simulates that season 100 times, and shows it all in a web app. AI Lab project.

## Course requirements and where they are covered

| # | Requirement | Where |
|---|---|---|
| 1 | Problem statement | This page (first paragraph): predict the result of a match before it is played |
| 2 | Dataset | Section "Data" below; features and target built in `src/2_features.py` |
| 3 | Preprocessing | `src/1_load_data.py` (missing values, duplicates, date parsing, sorting) and `src/2_features.py` (result letters H/D/A coded as 0/1/2; team names become Elo and form numbers instead of one-hot columns). Scaling: `StandardScaler` in `src/3_train.py`. Time-based train/test split: `src/3_train.py` |
| 4 | Exploratory data analysis | **Not covered in this simple version** (no plots yet). See the note at the end |
| 5 | Train 2 to 4 models | `src/3_train.py` (Logistic Regression, Random Forest); metrics in `src/4_evaluate.py` |
| 6 | Compare models, suggest the best | `src/4_evaluate.py` writes `results/comparison.csv`; discussion below |
| 7 | Web app | `app/streamlit_app.py` (Match Predictor and Season Table pages) |
| 8 | Submission | Source code on GitHub, dataset link below; report and demo are separate |

## How to run

```
pip install -r requirements.txt
python src/1_load_data.py     # 11 season files -> data/processed/matches.csv
python src/2_features.py      # Elo + last-5 form -> data/processed/features.csv
python src/3_train.py         # trains and saves both models in models/
python src/4_evaluate.py      # scores 2025-26 -> results/comparison.csv
python src/5_simulate.py      # 100 simulated seasons -> results/simulated_table.csv
streamlit run app/streamlit_app.py
```

Run every command from this folder.

## Method in four lines

- **Features** (all from matches played BEFORE the one being predicted): home and away Elo rating, their difference, and each team's average goals scored and conceded in its last 5 matches.
- **Split by time:** train on 2015-16 to 2024-25 (3,800 matches), test on 2025-26 (380 matches). Never a random split.
- **Models:** Logistic Regression on 3 features (`elo_diff`, `home_gf_5`, `away_gf_5`); Random Forest with default settings on all 7.
- **Simulation:** for each of the 380 fixtures the Logistic Regression gives home/draw/away probabilities, one outcome is drawn at random, points are added up, and this is repeated 100 times.

## Results (2025-26 test season, 380 matches)

| Model | Accuracy | Log loss (lower is better) | Draws predicted |
|---|---|---|---|
| Logistic Regression | 48.7% | 1.028 | 0 |
| Random Forest | 45.8% | 1.106 | 38 |
| Always home win | 42.6% | n/a | 0 |

The Logistic Regression is the suggested model: it has the lower log loss and the higher accuracy, and it needs only three numbers. The Random Forest with default settings memorises the training seasons (log loss 0.25 on the training matches against 1.11 on the test season), so its probabilities are too extreme; on the test season it is even worse than always predicting the usual outcome shares (log loss 1.084). With 380 matches, accuracy has a noise of about 2.5 points, so the gap between 48.7% and 45.8% is suggestive, not proof; the log-loss gap (0.077) is the clearer sign. The Logistic Regression never predicts a draw; the Random Forest predicts 38 and gets 12 of them right. Draws were 27% of 2025-26 matches and are rarely the single most likely outcome, so both models are weak on them.

**Limits to be aware of**
- The season simulation takes each fixture's saved pre-match features. For 2025-26 those were built from the real results of earlier matches, so the simulation is a random replay of the real season, not a pre-season forecast, and it is sharper than a true forecast would be.
- The Match Predictor page uses each team's state at the END of 2025-26, so it answers "what would the models say today", not "what would they have said before that match".
- No exploratory data analysis is included in this simple version (requirement 4).

## Data

Match results from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (Premier League, one CSV per season, 2015-16 to 2025-26; file `E0.csv` of each season). Columns used: `Date`, `HomeTeam`, `AwayTeam`, `FTHG` and `FTAG` (full-time goals), `FTR` (full-time result).
