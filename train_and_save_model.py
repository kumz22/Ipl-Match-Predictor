import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("matches.csv")

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

# Head-to-head win rate
head2head = {}
h2h_rate = []

for _, row in df.iterrows():
    t1, t2, winner = row['team1'], row['team2'], row['winner']
    pair = tuple(sorted([t1, t2]))
    head2head.setdefault(pair, {t1: 0, t2: 0, 'total': 0})

    t1_h2h = head2head[pair][t1] / head2head[pair]['total'] if head2head[pair]['total'] else 0.5
    h2h_rate.append(t1_h2h)

    if winner == t1:
        head2head[pair][t1] += 1
    elif winner == t2:
        head2head[pair][t2] += 1
    head2head[pair]['total'] += 1

df['t1_h2h_winrate'] = h2h_rate

# Home ground indicator
home_venues = {
    'Chennai Super Kings': 'MA Chidambaram Stadium, Chepauk',
    'Mumbai Indians': 'Wankhede Stadium',
    'Royal Challengers Bangalore': 'M Chinnaswamy Stadium',
    'Kolkata Knight Riders': 'Eden Gardens',
    'Rajasthan Royals': 'Sawai Mansingh Stadium',
    'Delhi Capitals': 'Arun Jaitley Stadium',
    'Kings XI Punjab': 'Punjab Cricket Association Stadium, Mohali',
    'Sunrisers Hyderabad': 'Rajiv Gandhi International Stadium, Uppal',
    'Lucknow Super Giants': 'BRSABV Ekana Cricket Stadium',
    'Gujarat Titans': 'Narendra Modi Stadium'
}

df['team1_home'] = df.apply(lambda row: 1 if home_venues.get(row['team1']) == row['venue'] else 0, axis=1)

# Final features and target
features = ['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_bin',
            'venue_enc', 'match_year', 'team1_win_pct', 'team2_win_pct',
            't1_h2h_winrate', 'team1_home']
X = df[features]
y = df['winner_enc']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Model trained. Accuracy: {model.score(X_test, y_test):.2f}")

# Save model and encoders
with open("ipl_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("team_encoder.pkl", "wb") as f:
    pickle.dump(team_encoder, f)

with open("venue_encoder.pkl", "wb") as f:
    pickle.dump(venue_encoder, f)

print("✅ All files saved: ipl_model.pkl, team_encoder.pkl, venue_encoder.pkl")
