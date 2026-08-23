# streamlit app - pick two teams, get a win probability
# run with: streamlit run app/streamlit_app.py

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import MODEL_PATH, TEAM_LATEST_STATS_PATH

st.set_page_config(page_title="NBA Win Predictor", page_icon="🏀", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_team_stats():
    return pd.read_csv(TEAM_LATEST_STATS_PATH).sort_values("TEAM_NAME").reset_index(drop=True)


def build_feature_row(home_stats, away_stats, feature_columns):
    row = {
        "SEASON_WIN_PCT_HOME": home_stats["SEASON_WIN_PCT"],
        "SEASON_WIN_PCT_AWAY": away_stats["SEASON_WIN_PCT"],
        "RECENT_FORM_HOME": home_stats["RECENT_FORM"],
        "RECENT_FORM_AWAY": away_stats["RECENT_FORM"],
        "AVG_PTS_SCORED_HOME": home_stats["AVG_PTS_SCORED"],
        "AVG_PTS_SCORED_AWAY": away_stats["AVG_PTS_SCORED"],
        "AVG_PTS_ALLOWED_HOME": home_stats["AVG_PTS_ALLOWED"],
        "AVG_PTS_ALLOWED_AWAY": away_stats["AVG_PTS_ALLOWED"],
    }
    row["WIN_PCT_DIFF"] = row["SEASON_WIN_PCT_HOME"] - row["SEASON_WIN_PCT_AWAY"]
    row["FORM_DIFF"] = row["RECENT_FORM_HOME"] - row["RECENT_FORM_AWAY"]
    row["PTS_SCORED_DIFF"] = row["AVG_PTS_SCORED_HOME"] - row["AVG_PTS_SCORED_AWAY"]
    row["PTS_ALLOWED_DIFF"] = row["AVG_PTS_ALLOWED_HOME"] - row["AVG_PTS_ALLOWED_AWAY"]
    return pd.DataFrame([row])[feature_columns]


def main():
    st.title("🏀 Which team actually wins?")
    st.caption("Logistic regression trained on NBA games from 2019-2024. Pick two teams below.")

    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle["model_name"]

    teams = load_team_stats()
    team_names = teams["TEAM_NAME"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", team_names, index=team_names.index("Boston Celtics") if "Boston Celtics" in team_names else 0)
    with col2:
        away_default = "Los Angeles Lakers" if "Los Angeles Lakers" in team_names else team_names[1]
        away_team = st.selectbox("Away team", team_names, index=team_names.index(away_default))

    if home_team == away_team:
        st.warning("pick two different teams")
        return

    home_stats = teams[teams["TEAM_NAME"] == home_team].iloc[0]
    away_stats = teams[teams["TEAM_NAME"] == away_team].iloc[0]

    X = build_feature_row(home_stats, away_stats, feature_columns)
    X_input = scaler.transform(X) if scaler is not None else X
    home_win_prob = model.predict_proba(X_input)[0, 1]

    st.divider()
    st.subheader("Prediction")

    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} (home)", f"{home_win_prob:.1%}")
    c2.metric(f"{away_team} (away)", f"{1 - home_win_prob:.1%}")
    st.progress(float(home_win_prob))

    winner = home_team if home_win_prob >= 0.5 else away_team
    st.success(f"Model picks: {winner}")

    st.divider()
    st.subheader("Stats behind the prediction")
    compare_df = pd.DataFrame({
        "Stat": ["Season win %", "Last 5 form", "Avg points scored", "Avg points allowed"],
        home_team: [
            f"{home_stats['SEASON_WIN_PCT']:.1%}",
            f"{home_stats['RECENT_FORM']:.1%}",
            f"{home_stats['AVG_PTS_SCORED']:.1f}",
            f"{home_stats['AVG_PTS_ALLOWED']:.1f}",
        ],
        away_team: [
            f"{away_stats['SEASON_WIN_PCT']:.1%}",
            f"{away_stats['RECENT_FORM']:.1%}",
            f"{away_stats['AVG_PTS_SCORED']:.1f}",
            f"{away_stats['AVG_PTS_ALLOWED']:.1f}",
        ],
    })
    st.table(compare_df.set_index("Stat"))

    with st.expander("About this model"):
        st.write(f"""
        - Model: {model_name.replace('_', ' ').title()}
        - Trained on regular season games from 2019-20 through 2023-24
        - Features are all computed before the game happens (win %, last 5 form, scoring avgs) so nothing leaks
        - Note: team stats here are each team's most recent snapshot from the training data, not live current-season numbers
        """)


if __name__ == "__main__":
    main()
