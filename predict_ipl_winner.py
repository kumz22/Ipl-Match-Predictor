import pickle
import pandas as pd

def predict_winner(team1, team2, venue, toss_winner, toss_decision, match_year):
    # Load the saved model and encoders
    model = pickle.load(open("ipl_model.pkl", "rb"))
    team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
    venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))

    # Encode inputs
    team1_enc = team_encoder.transform([team1])[0]
    team2_enc = team_encoder.transform([team2])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_dec = 1 if toss_decision == 'bat' else 0

    # Create input feature vector
    input_features = pd.DataFrame({
        'team1_enc': [team1_enc],
        'team2_enc': [team2_enc],
        'venue_enc': [venue_enc],
        'toss_winner_enc': [toss_enc],
        'toss_decision_bin': [toss_dec],
        'match_year': [match_year],
        'team1_win_pct': [0.5],  # Default win percentage
        'team2_win_pct': [0.5]   # Default win percentage
    })

    # Predict winner
    winner_enc = model.predict(input_features)[0]
    winner = team_encoder.inverse_transform([winner_enc])[0]

    return winner

if __name__ == "__main__":
    # Example usage
    team1 = "Mumbai Indians"
    team2 = "Chennai Super Kings"
    venue = "Wankhede Stadium"
    toss_winner = "Mumbai Indians"
    toss_decision = "bat"
    match_year = 2024

    predicted_winner = predict_winner(team1, team2, venue, toss_winner, toss_decision, match_year)
    print(f"Predicted Winner: {predicted_winner}")
