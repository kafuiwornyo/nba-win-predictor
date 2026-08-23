# NBA Win Predictor

Pick two NBA teams and get a win probability for the home team, plus the stats behind the prediction.

I built this to practice the full ML pipeline start to finish - pulling data, building features, training/comparing models, and wrapping it in something you can actually click around in instead of just a script.

## What it does

Given a home team and away team, predicts the probability the home team wins based on:
- season win % so far
- win % over their last 5 games
- average points scored
- average points allowed

All of these are computed as of *before* each game, so the model isn't cheating by seeing the result of the game it's predicting.

## Data

Used a public dataset of NBA box scores from 2010-2024 (found it on GitHub, no API key needed): https://github.com/NocturneBear/NBA-Data-2010-2024

Only trained on the last 5 seasons (2019-20 to 2023-24) since old rosters from 2010 aren't really useful for predicting games now.

## Model

Compared logistic regression against a random forest. Split the data by date (train on earlier games, test on later ones) instead of a random split, because randomly splitting time series data lets the model "see the future" during training which isn't realistic.

Results on the holdout set (most recent ~1,150 games):

| Model | Accuracy | AUC |
|---|---|---|
| always pick home team | 54.3% | - |
| logistic regression | 59.6% | 0.666 |
| random forest | 59.4% | 0.652 |

Logistic regression won by a hair and is what the app uses.

Home court advantage alone gets you to 54%, so beating that by 5+ points with 4 pretty simple features felt like a decent sanity check that the features actually mean something. Points allowed/scored ended up mattering the most, more than recent form.

## Running it

```
pip install -r requirements.txt

python src/fetch_data.py
python src/build_features.py
python src/train_model.py

streamlit run app/streamlit_app.py
```

## Files

```
data/              raw csv + generated features/model
src/config.py      paths and settings
src/fetch_data.py       downloads the dataset
src/build_features.py   builds the win/loss/stats features per matchup
src/train_model.py      trains + compares the two models
app/streamlit_app.py    the dashboard
```

## Things I ran into / learned

- First version of my rolling averages didn't shift by 1 game and accuracy came out like 90%+ which should've been a red flag immediately - the model was basically seeing the answer
- Random train/test split gave a much better looking number than the date-based split, but the date-based one is the honest one for this kind of data
- balldontlie.io (the usual free NBA API) now requires signing up for an API key, so I just used a pre-scraped CSV from GitHub instead to keep things simple

## Ideas if I keep working on this

- add rest days between games as a feature
- try xgboost instead of random forest
- track how it does on actual upcoming games instead of just historical test data
