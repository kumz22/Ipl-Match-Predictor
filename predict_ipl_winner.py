import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data
df = pd.read_csv("matches.csv")

# Clean date and extract year
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['team1', 'team2', 'winner', 'toss_winner', 'venue', 'date'])

df = df.sort_values('date')  # chronological

# Encode teams and venues
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

# ---- New Features ----

# 1. Win percentage
win_counts = {}
match_counts = {}
win_pct = []

for i, row in df.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    winner = row['winner']
    
    # Initialize counters
    for team in [t1, t2]:
        match_counts.setdefault(team, 0)
        win_counts.setdefault(team, 0)
    
    # Calculate current win rate before match
    if match_counts[t1] == 0:
        t1_win_pct = 0.5
    else:
        t1_win_pct = win_counts[t1] / match_counts[t1]
        
    if match_counts[t2] == 0:
        t2_win_pct = 0.5
    else:
        t2_win_pct = win_counts[t2] / match_counts[t2]
    
    win_pct.append([t1_win_pct, t2_win_pct])

    # Update after match
    match_counts[t1] += 1
    match_counts[t2] += 1
    if winner == t1:
        win_counts[t1] += 1
    elif winner == t2:
        win_counts[t2] += 1

df[['team1_win_pct', 'team2_win_pct']] = pd.DataFrame(win_pct, index=df.index)

# 2. Head-to-head win rate
head2head = {}
h2h_pct = []

for i, row in df.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    winner = row['winner']
    pair = tuple(sorted([t1, t2]))
    
    if pair not in head2head:
        head2head[pair] = {t1: 0, t2: 0, 'total': 0}

    record = head2head[pair]
    
    # Current win rate before match
    if record['total'] == 0:
        t1_h2h = 0.5
    else:
        t1_h2h = record[t1] / record['total']
    
    h2h_pct.append(t1_h2h)

    # Update record
    if winner == t1:
        record[t1] += 1
    elif winner == t2:
        record[t2] += 1
    record['total'] += 1

df['t1_h2h_winrate'] = h2h_pct

# 3. Home advantage
home_teams = {
    'Chennai Super Kings': 'MA Chidambaram Stadium, Chepauk',
    'Mumbai Indians': 'Wankhede Stadium',
    'Royal Challengers Bangalore': 'M Chinnaswamy Stadium',
    'Kolkata Knight Riders': 'Eden Gardens',
    'Rajasthan Royals': 'Sawai Mansingh Stadium',
    'Delhi Capitals': 'Arun Jaitley Stadium',
    'Kings XI Punjab': 'Punjab Cricket Association Stadium, Mohali',
    'Sunrisers Hyderabad': 'Rajiv Gandhi International Stadium',
    'Lucknow Super Giants': 'BRSABV Ekana Cricket Stadium',
    'Gujarat Titans': 'Narendra Modi Stadium'
}

def is_home(team, venue):
    return int(home_teams.get(team, '') == venue)

df['team1_home'] = df.apply(lambda row: is_home(row['team1'], row['venue']), axis=1)

# ----------------------------

# Drop any remaining NA
df = df.dropna()

# Feature set
X = df[['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_bin',
        'venue_enc', 'match_year', 'team1_win_pct', 'team2_win_pct',
        't1_h2h_winrate', 'team1_home']]
y = df['winner_enc']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print(f"✅ Model Accuracy: {model.score(X_test, y_test):.2f}")

# ---- Prediction ----
print("\nAvailable Teams:")
for i, team in enumerate(team_encoder.classes_):
    print(f"{i}: {team}")
t1 = int(input("\nEnter Team 1 (number): "))
t2 = int(input("Enter Team 2 (number): "))
toss = int(input("Enter Toss Winner (number): "))
decision = input("Toss Decision (bat/field): ").strip().lower()
decision_bin = 1 if decision == 'bat' else 0

print("\nAvailable Venues:")
for i, venue in enumerate(venue_encoder.classes_):
    print(f"{i}: {venue}")
venue = int(input("\nEnter Venue (number): "))
year = int(input("Enter Season Year (e.g., 2023): "))

# Dummy values for new features (use average or defaults for now)
input_data = [[
    t1, t2, toss, decision_bin, venue, year,
    0.5, 0.5, 0.5, is_home(team_encoder.inverse_transform([t1])[0], venue_encoder.inverse_transform([venue])[0])
]]

pred = model.predict(input_data)[0]
print(f"\n🏏 Predicted Winner: {team_encoder.inverse_transform([pred])[0]}")
