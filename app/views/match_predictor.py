"""Page 1: Match Predictor. The user picks two teams; the saved models answer from each team's end-of-2025-26 state."""
import altair as alt
import pandas as pd
import streamlit as st

import styles
from predictor import CLASSIFIERS, AppDataError, Predictor
from styles import CYAN, GREEN, GREY, crest, html, tile, tile_row
from team_colors import CODES, COLORS


@st.cache_resource(show_spinner="Loading the saved models...")
def load_predictor():
    return Predictor()


def pcts(p):
    """Home/draw/away as whole percentages that always add up to 100. Rounding each value on its own can give 99 or 101,
    so round all three down, then give the missing points to the values that were rounded down the most."""
    raw = [v * 100 for v in p]
    whole = [int(v) for v in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - whole[i], reverse=True)[:100 - sum(whole)]:
        whole[i] += 1
    return [f"{w}%" for w in whole]


def matchup_card(home, away):
    def side(team, align):
        return (f'<div class="side" style="background:linear-gradient(135deg, {COLORS[team]}33, transparent)">'
                f'{crest(team, 96)}<div class="name">{team}</div><div class="note">{align}</div></div>')
    html(f'<div class="matchup">{side(home, "HOME")}<div class="vs">VS</div>{side(away, "AWAY")}</div>')


def probability_tiles(p):
    html(tile_row([tile(text, label, color) for text, label, color in zip(pcts(p), ["Home win", "Draw", "Away win"], [GREEN, GREY, CYAN])]))


def scoreboard(r):
    html(f'<div class="score">{CODES[r["home"]]} {r["gf_home"]:.1f} : {r["gf_away"]:.1f} {CODES[r["away"]]}'
         f'<small>Goals scored per match in each team\'s last 5 games (recent form, not a prediction)</small></div>')


def comparison_table(r):
    body = ""
    for key, name in CLASSIFIERS.items():
        cells = pcts(r["probabilities"][key])
        body += (f'<tr class="{"primary" if key == "logreg" else ""}"><td>{name}</td><td style="color:{GREEN}">{cells[0]}</td>'
                 f'<td style="color:{GREY}">{cells[1]}</td><td style="color:{CYAN}">{cells[2]}</td></tr>')
    html('<div class="table-wrap"><table class="pl-table"><tr><th>Model</th><th>P(Home)</th><th>P(Draw)</th><th>P(Away)</th></tr>' + body + '</table></div>')


def real_result_section(predictor, r):
    """What really happened in this fixture in 2025-26, next to what each model gave to that outcome."""
    real = predictor.real_result(r["home"], r["away"])
    st.markdown("### What really happened")
    if real is None:
        st.caption("This pairing was not played in 2025-26.")
        return
    names = ["Home win", "Draw", "Away win"]
    colors = [GREEN, GREY, CYAN]
    date = pd.to_datetime(real["date"]).strftime("%d %B %Y")
    html(f'<div class="score sm">{CODES[r["home"]]} {real["home_score"]} : {real["away_score"]} {CODES[r["away"]]}'
         f'<small>Real full-time result, {date}: <b style="color:{colors[real["outcome"]]}">{names[real["outcome"]]}</b></small></div>')
    given = {"Logistic Regression (primary)": r["probabilities"]["logreg"], "Random Forest": r["probabilities"]["random_forest"]}
    st.caption("Chance each model gave to the outcome that really happened:")
    tiles = [tile(pcts(p)[real["outcome"]], f"{name}<br>top pick: {names[int(p.argmax())]}", colors[real["outcome"]], small=True) for name, p in given.items()]
    html(tile_row(tiles, wrap=True))
    st.caption("Read this as a sanity check, not a test: the models here use each team's form at the END of 2025-26, which already includes "
               "this very match, and the real score is never an input to any model. The fair test of the models is the 2025-26 comparison table in the README.")


def probability_chart(r):
    rows = [{"model": CLASSIFIERS[key], "outcome": outcome, "prob": value * 100}
            for key in CLASSIFIERS for outcome, value in zip(["Home win", "Draw", "Away win"], r["probabilities"][key])]
    chart = (alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4)
             .encode(y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelColor="#FFF", labelFontSize=14, labelLimit=400)),
                     x=alt.X("prob:Q", title="Probability (%)", axis=alt.Axis(labelColor="#FFF", titleColor=GREY, gridColor="#4a1450")),
                     yOffset="outcome:N",
                     color=alt.Color("outcome:N", legend=alt.Legend(title=None, orient="bottom", labelColor="#FFF"),
                                     scale=alt.Scale(domain=["Home win", "Draw", "Away win"], range=[GREEN, GREY, CYAN])),
                     tooltip=["model", "outcome", alt.Tooltip("prob:Q", format=".1f")])
             .properties(height=260, background="transparent").configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")


styles.hero("Match Predictor", "Premier League 2025-26 - end-of-season form")
styles.nav_links()
try:
    predictor = load_predictor()
except AppDataError as error:
    st.error(f"{error}. Please run the scripts in src/ first.")
    st.stop()

st.caption("This page shows what the saved models would predict for a hypothetical match RIGHT NOW, using each team's Elo rating "
           "and goal form as of its last 2025-26 match. It is not a forecast of a real upcoming fixture.")
teams = predictor.teams
left, middle, right = st.columns([3, 1, 3])
home_team = left.selectbox("Home team", teams, index=teams.index("Liverpool"), key="home_team")
middle.markdown("<div style='text-align:center;font-weight:800;font-size:24px;padding-top:34px;color:#04F5FF'>VS</div>", unsafe_allow_html=True)
away_options = [t for t in teams if t != home_team]
away_team = right.selectbox("Away team", away_options, index=away_options.index("Bournemouth") if "Bournemouth" in away_options else 0, key="away_team")

result = predictor.predict(home_team, away_team)
matchup_card(home_team, away_team)
st.write("")
st.markdown("### Win / draw / loss (Logistic Regression, the primary model)")
probability_tiles(result["probabilities"]["logreg"])
scoreboard(result)
st.markdown("### Both models side by side")
comparison_table(result)
st.markdown("### Probabilities by model")
probability_chart(result)

real_result_section(predictor, result)

with st.expander("Match context"):
    favourite = home_team if result["elo_diff"] > 0 else away_team
    gap = abs(result["elo_diff"])
    st.write(f"Elo: {home_team} {result['elo_home']:.0f}, {away_team} {result['elo_away']:.0f}. "
             f"**{favourite}** is rated higher by {gap:.0f} Elo points. Both models use this gap as an input, together with each team's "
             f"recent goals scored (the Random Forest also uses goals conceded and the two ratings).")
with st.expander("How this works"):
    st.write("These are the two models the project built: a Logistic Regression and a Random Forest. They look at each team's Elo rating "
             "(a strength number that goes up after wins) and its goals in the last 5 matches. On the 2025-26 test season the Logistic "
             "Regression was right about 49% of the time; the Random Forest did worse, and neither predicts draws well. "
             "Treat every number here as a rough probability, not a certainty.")
styles.disclaimer()
