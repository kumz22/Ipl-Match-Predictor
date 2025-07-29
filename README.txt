# 🏏 IPL Match Winner Predictor (Machine Learning Project)

This project is a machine learning-based IPL (Indian Premier League) match winner predictor. It uses historical match data and team performance metrics to predict the winning team of upcoming matches.

---

## 📁 Project Structure


---

## 🚀 Features

- Preprocessing of IPL match data.
- Feature engineering (win percentages, head-to-head stats, toss decision impact, home advantage).
- Model training using RandomForest or XGBoost.
- Streamlit web app for user-friendly prediction interface.
- Deployment-ready structure.

---

## 🔧 Requirements

Install the following Python libraries:

```bash
pip install -r requirements.txt


🧠 How the Model Works
Encodes categorical data (team names, venue, toss decision).

Adds new features like:

Team win percentage

Head-to-head win rate

Home ground advantage

Trains a classification model to predict the winning team.

The trained model is saved as ipl_model.pkl.

💻 How to Run
🔹 1. Train the Model (Optional)
bash
Copy
Edit
python model_train.py
🔹 2. Run the Streamlit App
bash
Copy
Edit
streamlit run ipl_predictor_ui.py
Then open the URL shown in your terminal (usually http://localhost:8501).

📊 Example Prediction
You'll input:

Team 1 and Team 2

Toss winner

Toss decision

Venue

Match year

The model will return the predicted winner.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙌 Acknowledgements
Kaggle IPL Dataset

Scikit-learn Documentation

Streamlit Docs

👤 Author
Your Name – Kumaran E

⭐ If you found this project helpful, feel free to star the repository!