# ⚽ Intelligent EPL Match Insight Dashboard

An interactive Streamlit dashboard that combines machine learning with fan sentiment analysis to predict English Premier League (EPL) match outcomes — and explain **why**.

---

## 🎯 Features

- 🔮 **Match Outcome Prediction**  
  Predicts the likely result (Home Win / Draw / Away Win) using a Random Forest model trained on 5 seasons of EPL match data.

- 📊 **Team Performance Stats**  
  Auto-loads average pre-match stats (shots, corners, fouls, etc.) for each team to simulate realistic matchups.

- 🧠 **Feature-Based Explanations**  
  Shows which game stats had the strongest influence on each prediction.

- 🗣️ **Fan Sentiment Analysis**  
  Analyzes thousands of tweets using VADER to gauge how fans are feeling about each team.

- 💬 **Match Vibe Summary**  
  Provides a one-line narrative comparing the optimism or caution between both fanbases.

---

## 🧪 How It Works

- **Model:** Random Forest Classifier from `scikit-learn`
- **NLP:** VADER Sentiment Analyzer from `nltk`
- **Visualization:** Built with `Streamlit`, `matplotlib`, and `pandas`

---

## 📁 Project Structure

```
sports-analytics-dashboard/
├── app.py               # Main Streamlit app
├── data/                # EPL match CSVs + tweet dataset
├── models/              # Saved ML model (.pkl)
├── scripts/             # Training + sentiment analysis scripts
│   ├── match_preprocessing.py
│   ├── train_match_model.py
│   ├── sentiment_analysis.py
│   └── sentiment_preprocessing.py
├── utils/               # Utility functions
│   ├── team_stats.py    # Computes average stats per team
│   └── feature_importance.py # Shows model feature weights
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 How to Run It

1. **Clone the repo**  
   ```bash
   git clone https://github.com/HenryFowobaje/sports-analytics-dashboard.git
   cd sports-analytics-dashboard
   ```

2. **Create a virtual environment & install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the dashboard**  
   ```bash
   streamlit run app.py
   ```

---

## 📊 Data Sources

- **Football-Data.co.uk** — EPL match statistics
- **Twitter dataset** — EPL fan tweets (2020-2021 window)


---

## 📌 Future Improvements

- Add form-over-time performance graphs
- Include real-time tweet scraping with Twitter API
- Deploy publicly on Streamlit Cloud