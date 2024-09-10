import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV

# load match data
matches = pd.read_csv("matches.csv")

# convert 'date' to datetime format
matches["date"] = pd.to_datetime(matches["date"])

# map venue to home/away as 1 or 0
matches["home/away"] = matches["venue"].apply(lambda x: 1 if x == 'Home' else 0)  # 1 for home, 0 for away

# encode opponent column to numeric codes
matches["opponent_code"] = matches["opponent"].astype('category').cat.codes  # encoding opponent teams

# extract the hour from 'time' column
matches["hour"] = matches["time"].str.extract(r'(\d{1,2}):').astype(int)  # extract hour from time

# get the day of the week from 'date' column
matches["day_of_week"] = matches["date"].dt.dayofweek  # feature: day of the week

# target variable is whether the match was a win (1) or not (0)
matches["target"] = (matches["result"] == "W").astype(int)

# predictors for the model
predictors = ["home/away", "opponent_code", "hour", "day_of_week"]

# create rolling averages for recent form (last 3 matches)
def rolling_averages(group, cols, new_cols):
    # sort matches by date first
    group = group.sort_values("date")
    # rolling mean over last 3 games, excluding the current one
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    # assign to new columns
    group[new_cols] = rolling_stats
    # drop rows without enough data for rolling averages
    group = group.dropna(subset=new_cols)
    return group

# columns to use for performance stats
performance_cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
rolling_cols = [f"{col}_rolling" for col in performance_cols]

# apply rolling averages for each team
matches_rolling = matches.groupby("team").apply(lambda group: rolling_averages(group, performance_cols, rolling_cols))
matches_rolling.reset_index(drop=True, inplace=True)

# function to split rolling stats into home and away categories
def adjust_rolling_averages(df, is_home=True):
    for col in rolling_cols:
        if is_home:
            df[f'home_{col}'] = df[col]
        else:
            df[f'away_{col}'] = df[col]
    return df

# split rolling averages by home/away
matches_rolling = adjust_rolling_averages(matches_rolling, is_home=True)
matches_rolling = adjust_rolling_averages(matches_rolling, is_home=False)

# split train and test data based on date
train_rolling = matches_rolling[matches_rolling["date"] < '2022-01-01']
test_rolling = matches_rolling[matches_rolling["date"] >= '2022-01-01']

# final predictor list includes non-rolling and rolling stats
rolling_predictors_home = [f'home_{col}' for col in rolling_cols]
rolling_predictors_away = [f'away_{col}' for col in rolling_cols]
all_predictors = predictors + rolling_predictors_home + rolling_predictors_away

# create Random Forest model
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# cross-validate the model and print accuracy and precision
accuracy_scores = cross_val_score(rf, train_rolling[all_predictors], train_rolling["target"], cv=5, scoring='accuracy')
precision_scores = cross_val_score(rf, train_rolling[all_predictors], train_rolling["target"], cv=5, scoring='precision')

# log the cross-validation results
print(f"Cross-Validation Accuracy: {accuracy_scores.mean()} (±{accuracy_scores.std()})")
print(f"Cross-Validation Precision: {precision_scores.mean()} (±{precision_scores.std()})")

# train Random Forest model using the entire training set
rf.fit(train_rolling[all_predictors], train_rolling["target"])

# predict test set results and evaluate accuracy/precision
preds_rolling = rf.predict(test_rolling[all_predictors])
accuracy_rolling = accuracy_score(test_rolling["target"], preds_rolling)
precision_rolling = precision_score(test_rolling["target"], preds_rolling)

# log the test set performance
print(f"Rolling Data - Accuracy: {accuracy_rolling}")
print(f"Rolling Data - Precision: {precision_rolling}")

# prepare future matches for prediction
future_matches = pd.read_csv("future_matches.csv")

# convert 'Date' to datetime and extract features
future_matches["date"] = pd.to_datetime(future_matches["Date"], format='%d/%m/%Y %H:%M')
future_matches["hour"] = future_matches["date"].dt.hour
future_matches["day_of_week"] = future_matches["date"].dt.dayofweek

# map home/away for future matches
future_matches["home/away"] = future_matches["Location"].apply(lambda x: 1 if "Home" in x else 0)

# map team names to numeric codes
team_codes = pd.concat([matches["team"], matches["opponent"]]).unique()
team_mapping = {team: code for code, team in enumerate(team_codes)}

# handle cases where teams are not found in historical data
future_matches["home_team_code"] = future_matches["Home Team"].map(team_mapping).fillna(-1)
future_matches["away_team_code"] = future_matches["Away Team"].map(team_mapping).fillna(-1)
future_matches["opponent_code"] = future_matches["away_team_code"]

# define non-rolling predictors
non_rolling_predictors = ["home/away", "opponent_code", "hour", "day_of_week"]

# function to get latest rolling averages for teams
def get_latest_rolling_averages(team, historical_data, rolling_cols):
    team_data = historical_data[historical_data["team"] == team]
    if not team_data.empty:
        latest_averages = team_data.iloc[-1][rolling_cols]  # take most recent averages
        return latest_averages
    else:
        return pd.Series([0] * len(rolling_cols), index=rolling_cols)  # if no history, default to 0

# apply rolling averages for future matches
for index, row in future_matches.iterrows():
    home_team_rolling = get_latest_rolling_averages(row["Home Team"], matches_rolling, rolling_cols)
    away_team_rolling = get_latest_rolling_averages(row["Away Team"], matches_rolling, rolling_cols)

    # assign the rolling averages to the future matches DataFrame
    for col in rolling_cols:
        future_matches.loc[index, f"home_{col}"] = home_team_rolling[col]
        future_matches.loc[index, f"away_{col}"] = away_team_rolling[col]

# fill any missing values with 0
future_matches.fillna(0, inplace=True)

# predict outcomes for future matches
all_future_predictors = non_rolling_predictors + rolling_predictors_home + rolling_predictors_away
future_predictions = rf.predict(future_matches[all_future_predictors])

# store predictions in the DataFrame
future_matches["Predicted Result"] = future_predictions

# save future match predictions to a CSV
future_matches[["Date", "Home Team", "Away Team", "Predicted Result"]].to_csv("future_predictions.csv", index=False)
print(f"Predictions saved to future_predictions.csv")
