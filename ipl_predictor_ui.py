import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model and encoders
model = pickle.load(open("ipl_model.pkl", "rb"))
team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))

# Map teams to their home stadiums
home_stadiums = {
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

def get_home_status(team1, venue):
    return int(home_stadiums.get(team1, '') == venue)

# UI starts here
st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")

st.markdown("""
Select match details below to predict the winning team using our trained machine learning model.
""")

# Input fields
team1 = st.selectbox("Team 1", team_encoder.classes_)
team2 = st.selectbox("Team 2", [team for team in team_encoder.classes_ if team != team1])
venue = st.selectbox("Venue", venue_encoder.classes_)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ['bat', 'field'])
match_year = st.number_input("Match Year", value=2024, min_value=2008, max_value=2025)

if st.button("Predict Winner"):
    try:
        # Encode inputs
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        toss_enc = team_encoder.transform([toss_winner])[0]
        venue_enc = venue_encoder.transform([venue])[0]
        toss_dec = 1 if toss_decision == 'bat' else 0
        is_home = get_home_status(team1, venue)

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
        prediction = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        predicted_winner = team_encoder.inverse_transform([prediction])[0]

        # Display results
        st.success(f"Predicted Winner: {predicted_winner}")
        st.write(f"Prediction Confidence: {max(prediction_proba) * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred: {e}")
