import streamlit as st
import pandas as pd
import joblib
from utils.team_stats import load_team_averages
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("models/match_predictor.pkl")
team_stats = load_team_averages()
sentiment_df = pd.read_csv("data/team_sentiment_labeled.csv")

# Page config
st.set_page_config(page_title="EPL Match Predictor", layout="centered")
st.title("⚽ EPL Match Insight Dashboard")
st.markdown("Select two Premier League teams to view predictions, fan sentiment, and team stats.")

teams = sorted(list(team_stats.keys()))
home_team = st.selectbox("🏠 Select Home Team", teams)
away_team = st.selectbox("🚗 Select Away Team", [t for t in teams if t != home_team])

# Sentiment fetcher
def get_team_sentiment(team):
    team_data = sentiment_df[sentiment_df["team"] == team]
    return team_data["sentiment_label"].value_counts()

# Fan summary formatter
def get_sentiment_summary(sentiment_series):
    total = sentiment_series.sum()
    if total == 0:
        return "No sentiment data available for this team."
    pos = sentiment_series.get("positive", 0)
    neu = sentiment_series.get("neutral", 0)
    neg = sentiment_series.get("negative", 0)
    return f"**{(pos/total)*100:.1f}%** positive · **{(neu/total)*100:.1f}%** neutral · **{(neg/total)*100:.1f}%** negative"

# Predict button
if st.button("🔮 Predict Match Result"):
    # Build input vector
    home = team_stats[home_team]
    away = team_stats[away_team]

    input_data = {
        "HS": home["HS"], "AS": away["AS"],
        "HST": home["HST"], "AST": away["AST"],
        "HC": home["HC"], "AC": away["AC"],
        "HF": home["HF"], "AF": away["AF"],
        "HY": home["HY"], "AY": away["AY"],
        "HR": home["HR"], "AR": away["AR"]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df).max()

    result_map = {1: "🏠 Home Win", 0: "🤝 Draw", -1: "🚗 Away Win"}
    st.header(f"📊 Predicted Outcome: {result_map[prediction]}")
    st.caption(f"Confidence: **{confidence:.2%}**")

    # Team stat summary
    with st.expander("📋 Team Stats Overview"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{home_team} (Home)")
            for k, v in home.items():
                stat_label = {
                    "HS": "Total Shots",
                    "HST": "Shots on Target",
                    "HC": "Corners",
                    "HF": "Fouls",
                    "HY": "Yellow Cards",
                    "HR": "Red Cards"
                }.get(k, k)
                st.write(f"**{stat_label}**: {round(v, 2)}")

        with col2:
            st.subheader(f"{away_team} (Away)")
            for k, v in away.items():
                stat_label = {
                    "AS": "Total Shots",
                    "AST": "Shots on Target",
                    "AC": "Corners",
                    "AF": "Fouls",
                    "AY": "Yellow Cards",
                    "AR": "Red Cards"
                }.get(k, k)
                st.write(f"**{stat_label}**: {round(v, 2)}")

    # Fan sentiment with breakdowns
    with st.expander("🗣️ What Fans Are Saying About This Match"):
        st.markdown(f"""
        Understanding fan sentiment gives a unique look into the emotional pulse of supporters.

        • **{home_team} fans** might reflect hope, frustration, or pure hype — which often mirrors locker-room vibes.  
        • **{away_team} fans** bring pressure, energy, or confidence that can shift narratives.

        Let's see how each fanbase is feeling ahead of this matchup:
        """)

        col1, col2 = st.columns(2)
        home_sentiment = get_team_sentiment(home_team)
        away_sentiment = get_team_sentiment(away_team)

        with col1:
            st.subheader(f"{home_team} Fans")
            st.bar_chart(home_sentiment)
            st.caption(get_sentiment_summary(home_sentiment))

        with col2:
            st.subheader(f"{away_team} Fans")
            if not away_sentiment.empty:
                st.bar_chart(away_sentiment)
                st.caption(get_sentiment_summary(away_sentiment))
            else:
                st.markdown("_No fan sentiment data available yet for this team._")

        # Match vibe insight
        if not away_sentiment.empty:
            home_total = home_sentiment.sum()
            away_total = away_sentiment.sum()
            home_pos = home_sentiment.get("positive", 0) / home_total if home_total else 0
            away_pos = away_sentiment.get("positive", 0) / away_total if away_total else 0

            if home_pos > away_pos + 0.1:
                vibe_summary = f"⚡ **{home_team} fans are noticeably more optimistic** heading into this match than {away_team} fans."
            elif away_pos > home_pos + 0.1:
                vibe_summary = f"⚡ **{away_team} fans seem more confident** going into this game than {home_team} fans."
            else:
                vibe_summary = f"🤝 **Both fanbases share similar levels of hope and caution**."

            st.markdown(f"### 🧠 Match Vibe: {vibe_summary}")

    # Model explainability
    with st.expander("🔍 Why This Prediction?"):
        feature_names = list(input_data.keys())
        importances = model.feature_importances_

        top_features = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(5)

        stat_labels = {
            "HST": "Home shots on target",
            "AST": "Away shots on target",
            "HS": "Home total shots",
            "AS": "Away total shots",
            "AF": "Away fouls",
            "HF": "Home fouls",
            "HC": "Home corners",
            "AC": "Away corners",
            "HY": "Home yellow cards",
            "AY": "Away yellow cards",
            "HR": "Home red cards",
            "AR": "Away red cards"
        }

        for _, row in top_features.iterrows():
            label = stat_labels.get(row["Feature"], row["Feature"])
            st.write(f"• **{label}** strongly influenced this prediction (weight: `{round(row['Importance'], 3)}`)")
