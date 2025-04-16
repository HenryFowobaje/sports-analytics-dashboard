import streamlit as st
import pandas as pd
import joblib
from utils.team_stats import load_team_averages
from utils.feature_importance import plot_feature_importance

# Load model
model = joblib.load("models/match_predictor.pkl")
team_stats = load_team_averages()

st.title("⚽ Intelligent EPL Match Predictor")
st.markdown("Select teams below to predict match outcome based on historical team stats.")

teams = sorted(list(team_stats.keys()))

# Team selection
home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", [t for t in teams if t != home_team])

if st.button("Predict Outcome"):
    # Get average stats for both teams
    home = team_stats[home_team]
    away = team_stats[away_team]

    # Construct feature input for the model
    input_data = {
        "HS": home["HS"],
        "AS": away["AS"],
        "HST": home["HST"],
        "AST": away["AST"],
        "HC": home["HC"],
        "AC": away["AC"],
        "HF": home["HF"],
        "AF": away["AF"],
        "HY": home["HY"],
        "AY": away["AY"],
        "HR": home["HR"],
        "AR": away["AR"]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max()

    result_map = {1: "🏠 Home Win", 0: "🤝 Draw", -1: "🚗 Away Win"}
    
    st.subheader(f"📊 Predicted Result: {result_map[prediction]}")
    st.caption(f"Model confidence: {prob:.2%}")

    # Show team stat summary
    with st.expander("📋 View Team Stats Used"):
        st.write("**Home Team Stats (avg)**")
        st.dataframe(pd.DataFrame([home]).round(2).T.rename(columns={0: home_team}))
        st.write("**Away Team Stats (avg)**")
        st.dataframe(pd.DataFrame([away]).round(2).T.rename(columns={0: away_team}))
    
    # Show top features that influence prediction
    with st.expander("🔍 Why this prediction? Top influencing features"):
        importance = model.feature_importances_
        feature_names = list(input_data.keys())
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(5)
        st.dataframe(feature_df.reset_index(drop=True))
