# NBA Win Predictor

this is a little app where you pick 2 nba teams and it tells you the % chance the home team wins. also shows the stats it used to decide.

this was my first real python project outside of class assignments so it was a lot of trial and error lol

## what it does

you pick a home team and away team and it gives you a probability based on:
- how many games they've won this season
- how they've done in their last 5 games
- how many points they usually score
- how many points they usually give up

i made sure it only uses stats from BEFORE the game happens, otherwise it would just be cheating basically (like if it already knew the score)

## where the data came from

i wanted to use the balldontlie API since everyone online said it was easy for beginners but turns out now you need an api key for it which i didnt wanna deal with. so instead i just found a csv on github that already has like 14 years of nba games in it and used that instead
link: https://github.com/NocturneBear/NBA-Data-2010-2024

i only used the last 5 seasons of data though bc using stuff from 2010 seemed kind of pointless, teams are way different now

## the model part

i tried two different models to see which one worked better - logistic regression and random forest. logistic regression ended up doing slightly better so thats what the app actually uses

one thing i learned (the hard way) is you cant just randomly split your data into train/test when its got dates involved. like i tried that first and got a way higher accuracy number but then realized thats kind of fake because it lets the model accidentally learn from "future" games. so i switched to just using the earlier games to train and the more recent games to test, which is more realistic

here's what i got:
- if you just always guess the home team wins: 54.3% right
- my logistic regression model: 59.6% right
- random forest: 59.4% right

so its not like insanely better than just guessing home team every time lol but it is better, and i think thats kind of the point, home court advantage is already a big factor by itself

## how to run it

pip install -r requirements.txt

python src/fetch_data.py
python src/build_features.py
python src/train_model.py

streamlit run app/streamlit_app.py

## whats in here

- data/ - the raw csv and stuff the scripts generate
- src/fetch_data.py - downloads the dataset
- src/build_features.py - turns the raw data into the stats the model uses
- src/train_model.py - trains the models and picks the better one
- app/streamlit_app.py - the actual app

## stuff that went wrong / things i learned

- my first attempt at this had a bug where the model could kind of see the answer without me realizing it (forgot to shift my rolling averages back by one game) and i got like 90% accuracy which should have been a red flag immediately lol, no way its that easy
- learned that splitting data by date matters a lot more than i thought for anything with a time component
- couldnt use the API i originally wanted to because of the key thing so just used a premade dataset instead, which honestly worked out fine

## things i might add later

- days of rest between games as a feature (teams playing back to back games probably play worse)
- try xgboost since i keep seeing people mention it
- test it on actual real upcoming games instead of just old data
- track how it does on actual upcoming games instead of just historical test data
