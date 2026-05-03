import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert date and drop missing
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['team1', 'team2', 'winner', 'toss_winner', 'venue', 'date'], inplace=True)
    df.sort_values('date', inplace=True)

    # Label encoding
    teams = pd.concat([df['team1'], df['team2'], df['winner'], df['toss_winner']]).unique()
    team_encoder = LabelEncoder().fit(teams)
    venue_encoder = LabelEncoder().fit(df['venue'].unique())

    df['team1_enc'] = team_encoder.transform(df['team1'])
    df['team2_enc'] = team_encoder.transform(df['team2'])
    df['winner_enc'] = team_encoder.transform(df['winner'])
    df['toss_winner_enc'] = team_encoder.transform(df['toss_winner'])
    df['venue_enc'] = venue_encoder.transform(df['venue'])
    df['match_year'] = df['date'].dt.year
    df['toss_decision_bin'] = df['toss_decision'].apply(lambda x: 1 if x == 'bat' else 0)

    # Team win % features
    win_counts = {}
    match_counts = {}
    team1_win_pct = []
    team2_win_pct = []

    for _, row in df.iterrows():
        t1, t2, winner = row['team1'], row['team2'], row['winner']
        for team in [t1, t2]:
            match_counts.setdefault(team, 0)
            win_counts.setdefault(team, 0)

        team1_win = win_counts[t1] / match_counts[t1] if match_counts[t1] else 0.5
        team2_win = win_counts[t2] / match_counts[t2] if match_counts[t2] else 0.5
        team1_win_pct.append(team1_win)
        team2_win_pct.append(team2_win)

        match_counts[t1] += 1
        match_counts[t2] += 1
        if winner == t1:
            win_counts[t1] += 1
        elif winner == t2:
            win_counts[t2] += 1

    df['team1_win_pct'] = team1_win_pct
    df['team2_win_pct'] = team2_win_pct

    return df, team_encoder, venue_encoder

def train_model(df):
    # Features and target
    X = df[['team1_enc', 'team2_enc', 'venue_enc', 'toss_winner_enc', 'toss_decision_bin', 'match_year', 'team1_win_pct', 'team2_win_pct']]
    y = df['winner_enc']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training with hyperparameter tuning
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Model Parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {best_model.score(X_train, y_train)}")
    print(f"Test Accuracy: {best_model.score(X_test, y_test)}")

    return best_model

def save_artifacts(model, team_encoder, venue_encoder):
    # Save model and encoders
    with open("ipl_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("team_encoder.pkl", "wb") as team_file:
        pickle.dump(team_encoder, team_file)
    with open("venue_encoder.pkl", "wb") as venue_file:
        pickle.dump(venue_encoder, venue_file)

if __name__ == "__main__":
    data_file = "matches.csv"
    df, team_encoder, venue_encoder = preprocess_data(data_file)
    model = train_model(df)
    save_artifacts(model, team_encoder, venue_encoder)
