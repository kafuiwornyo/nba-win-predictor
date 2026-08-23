# turns raw box score rows into matchup features for the model
#
# the raw csv has one row PER TEAM per game (so each game shows up
# twice, once for the home team once for away). i split those out,
# sort each team's games by date, and roll up pre-game stats:
#   - win pct so far this season
#   - win pct in last 5 games
#   - avg points scored/allowed so far
# then glue the home team's row and the away team's row back together
# into one row per game so the model can see both sides at once.
#
# important: everything uses .shift(1) before rolling/expanding so a
# game's own result never leaks into its own features. learned this
# the hard way after my first version had like 90% accuracy which was
# obviously wrong lol

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import RAW_CSV_PATH, FEATURES_PATH, TEAM_LATEST_STATS_PATH, SEASONS_TO_USE, RECENT_FORM_WINDOW


def load_raw():
    df = pd.read_csv(RAW_CSV_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[df["SEASON_YEAR"].isin(SEASONS_TO_USE)].copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.")
    df["WIN"] = (df["WL"] == "W").astype(int)
    return df


def add_rolling_team_features(df):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    group = df.groupby("TEAM_ID", group_keys=False)

    df["SEASON_WIN_PCT"] = group["WIN"].apply(lambda s: s.shift(1).expanding().mean())
    df["RECENT_FORM"] = group["WIN"].apply(lambda s: s.shift(1).rolling(RECENT_FORM_WINDOW, min_periods=1).mean())
    df["AVG_PTS_SCORED"] = group["PTS"].apply(lambda s: s.shift(1).expanding().mean())

    # need the opponent's score to get points allowed
    opp_pts = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(columns={"TEAM_ID": "OPP_ID", "PTS": "OPP_PTS"})
    df = df.merge(opp_pts, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_ID"]].copy()
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    group = df.groupby("TEAM_ID", group_keys=False)
    df["AVG_PTS_ALLOWED"] = group["OPP_PTS"].apply(lambda s: s.shift(1).expanding().mean())

    return df


def build_matchup_table(df):
    cols = [
        "GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",
        "IS_HOME", "WIN", "SEASON_WIN_PCT", "RECENT_FORM", "AVG_PTS_SCORED", "AVG_PTS_ALLOWED",
    ]
    slim = df[cols].copy()

    home = slim[slim["IS_HOME"]].drop(columns="IS_HOME")
    away = slim[~slim["IS_HOME"]].drop(columns="IS_HOME")

    merged = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON_YEAR"], suffixes=("_HOME", "_AWAY"))
    merged["HOME_WIN"] = merged["WIN_HOME"]

    merged["WIN_PCT_DIFF"] = merged["SEASON_WIN_PCT_HOME"] - merged["SEASON_WIN_PCT_AWAY"]
    merged["FORM_DIFF"] = merged["RECENT_FORM_HOME"] - merged["RECENT_FORM_AWAY"]
    merged["PTS_SCORED_DIFF"] = merged["AVG_PTS_SCORED_HOME"] - merged["AVG_PTS_SCORED_AWAY"]
    merged["PTS_ALLOWED_DIFF"] = merged["AVG_PTS_ALLOWED_HOME"] - merged["AVG_PTS_ALLOWED_AWAY"]

    # early season games don't have history yet, drop those
    merged = merged.dropna(subset=[
        "SEASON_WIN_PCT_HOME", "SEASON_WIN_PCT_AWAY",
        "RECENT_FORM_HOME", "RECENT_FORM_AWAY",
        "AVG_PTS_SCORED_HOME", "AVG_PTS_SCORED_AWAY",
        "AVG_PTS_ALLOWED_HOME", "AVG_PTS_ALLOWED_AWAY",
    ])

    return merged.sort_values("GAME_DATE").reset_index(drop=True)


def save_latest_team_stats(df):
    # this is for the app - each team's current stats so you can predict
    # a hypothetical matchup, not just replay a past game
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    group = df.groupby("TEAM_ID", group_keys=False)

    season_win_pct = group["WIN"].apply(lambda s: s.expanding().mean())
    recent_form = group["WIN"].apply(lambda s: s.rolling(RECENT_FORM_WINDOW, min_periods=1).mean())
    avg_pts_scored = group["PTS"].apply(lambda s: s.expanding().mean())
    avg_pts_allowed = group["OPP_PTS"].apply(lambda s: s.expanding().mean())

    df = df.assign(
        _wp=season_win_pct, _form=recent_form,
        _pts=avg_pts_scored, _allowed=avg_pts_allowed,
    )
    latest = df.groupby("TEAM_ID").tail(1)[
        ["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "_wp", "_form", "_pts", "_allowed"]
    ].rename(columns={
        "_wp": "SEASON_WIN_PCT", "_form": "RECENT_FORM",
        "_pts": "AVG_PTS_SCORED", "_allowed": "AVG_PTS_ALLOWED",
    })

    latest.to_csv(TEAM_LATEST_STATS_PATH, index=False)
    print(f"saved current stats for {len(latest)} teams -> {TEAM_LATEST_STATS_PATH}")


def main():
    raw = load_raw()
    print(f"loaded {len(raw)} rows for seasons {SEASONS_TO_USE}")

    with_features = add_rolling_team_features(raw)
    save_latest_team_stats(with_features)

    matchups = build_matchup_table(with_features)
    print(f"{len(matchups)} matchups after dropping early-season games with no history")
    print(f"home win rate: {matchups['HOME_WIN'].mean():.3f}")

    matchups.to_csv(FEATURES_PATH, index=False)
    print("saved ->", FEATURES_PATH)


if __name__ == "__main__":
    main()
