# Premier League Match Outcome Predictor

This project leverages machine learning to predict the outcomes of Premier League football matches using historical match data. The model is built using a combination of non-rolling and rolling features to capture team performance trends and recent form.

## Key Features

- **Data Preparation**: Encodes match information such as home/away status, opponent, match time, and day of the week.
- **Rolling Averages**: Incorporates rolling averages of key performance metrics (e.g., goals, shots, distance covered) over recent matches to account for team form.
- **Machine Learning Models**: Implements a Random Forest Classifier and XGBoost to predict match outcomes (win/loss) based on both static and rolling data.
- **Cross-Validation & Evaluation**: Uses cross-validation to evaluate model accuracy and precision, and tests predictions on unseen data.
- **Future Match Predictions**: Predicts the outcomes of upcoming matches by extracting relevant features and using the trained model to generate results.

## Overview

The project provides a robust approach to analyzing football match data and generating predictive insights into future match results. It is built using Python, Pandas for data manipulation, and Scikit-learn along with XGBoost for model training and evaluation.

## How it Works
1. The model takes in historical data with encoded match information.
2. Rolling averages are computed for team performance metrics.
3. The Random Forest and XGBoost models are trained on the data to predict match outcomes.
4. The model is tested using cross-validation, and its accuracy is evaluated.
5. Future matches are predicted using the trained model.

   
## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Random Forest


