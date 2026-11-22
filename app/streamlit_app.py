"""Premier League Prediction: the web app. Run from the final_project folder:  streamlit run app/streamlit_app.py

This file is the entry point: page settings, the shared look, the sidebar and the two pages (app/views/).
The app only SERVES things saved by src/ (models, team state, simulation results); it trains nothing.
"""
import streamlit as st

import styles

st.set_page_config(page_title="Premier League Prediction", page_icon="⚽", layout="wide", initial_sidebar_state="auto")
styles.inject_css()

with st.sidebar:
    st.markdown("### About")
    st.markdown("A university project (AI Lab). Two models (Logistic Regression and Random Forest) were trained on ten "
                "Premier League seasons (2015-16 to 2024-25) and tested on 2025-26. They use each team's Elo rating and recent goals.")
    st.markdown("### Limitations")
    st.markdown("- **No model predicts draws well.** Draws are about a quarter of matches and look like coin flips to every model.\n"
                "- **The simulation is a replay.** It uses each 2025-26 fixture's real pre-match form, so it is sharper than a true forecast.\n"
                "- **Simple on purpose.** Only 7 features and default settings: this is the small version of the project.")
    st.markdown("[Source code on GitHub](https://github.com/Ahmad24P3011/AI_FINAL_PROJECT)")

pages = [st.Page("views/match_predictor.py", title="Match Predictor", icon="⚽", default=True),
         st.Page("views/season_table.py", title="Season Table", icon="🏆")]
st.navigation(pages).run()
