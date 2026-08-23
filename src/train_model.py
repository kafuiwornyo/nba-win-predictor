# trains logistic regression + random forest and keeps whichever does
# better, then saves it. splits by date instead of randomly since this
# is time series data - if you shuffle randomly you basically let the
# model see the "future" during training which isnt realistic

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_PATH, MODEL_PATH

FEATURE_COLUMNS = [
    "SEASON_WIN_PCT_HOME", "SEASON_WIN_PCT_AWAY",
    "RECENT_FORM_HOME", "RECENT_FORM_AWAY",
    "AVG_PTS_SCORED_HOME", "AVG_PTS_SCORED_AWAY",
    "AVG_PTS_ALLOWED_HOME", "AVG_PTS_ALLOWED_AWAY",
    "WIN_PCT_DIFF", "FORM_DIFF", "PTS_SCORED_DIFF", "PTS_ALLOWED_DIFF",
]
TARGET_COLUMN = "HOME_WIN"


def chronological_split(df, test_frac=0.2):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"  {name:22s} acc={acc:.3f}  auc={auc:.3f}  brier={brier:.3f}")
    return acc, auc, brier


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["GAME_DATE"])
    train_df, test_df = chronological_split(df)
    print(f"train: {len(train_df)} games, test: {len(test_df)} games (split by date)")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    baseline_acc = accuracy_score(y_test, np.ones(len(y_test)))
    print(f"\nbaseline (just always pick the home team): acc={baseline_acc:.3f}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("comparing models:")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_scores = evaluate("logistic regression", logreg, X_test_scaled, y_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_scores = evaluate("random forest", rf, X_test, y_test)

    if rf_scores[1] >= logreg_scores[1]:
        best_name, best_model, needs_scaling = "random_forest", rf, False
    else:
        best_name, best_model, needs_scaling = "logistic_regression", logreg, True

    print("\nbest model:", best_name)

    bundle = {
        "model": best_model,
        "scaler": scaler if needs_scaling else None,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_name,
    }
    joblib.dump(bundle, MODEL_PATH)
    print("saved ->", MODEL_PATH)

    print("\nfeature importance:")
    if best_name == "random_forest":
        importances = sorted(zip(FEATURE_COLUMNS, best_model.feature_importances_), key=lambda x: -x[1])
        for feat, imp in importances:
            print(f"  {feat:22s} {imp:.3f}")
    else:
        coefs = sorted(zip(FEATURE_COLUMNS, best_model.coef_[0]), key=lambda x: -abs(x[1]))
        for feat, coef in coefs:
            print(f"  {feat:22s} {coef:+.3f}")


if __name__ == "__main__":
    main()
