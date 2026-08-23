import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

RAW_CSV_PATH = os.path.join(DATA_DIR, "regular_season_totals_2010_2024.csv")
RAW_CSV_URL = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/regular_season_totals_2010_2024.csv"

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")
TEAM_LATEST_STATS_PATH = os.path.join(DATA_DIR, "team_latest_stats.csv")

# just using last few seasons, old rosters don't really help predict now
SEASONS_TO_USE = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

RECENT_FORM_WINDOW = 5
