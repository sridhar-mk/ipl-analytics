# 🏏 IPL Player Performance Analytics System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey?logo=sqlite)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-RandomForest-green?logo=scikit-learn)
![Data](https://img.shields.io/badge/Data-260k%2B%20deliveries-orange)

An end-to-end data analytics system built on **16 seasons of real IPL data** (2008–2023).  
Combines advanced SQL analysis, Python EDA, machine learning, and an interactive Streamlit dashboard — framed as a real business problem a company like **Dream11 or CricBuzz** would solve.

---

## 🔴 Live Dashboard

👉 **[Open Live Dashboard](https://ipl-analytics.streamlit.app)**

---

## 🎯 Business Problem

> *"Which players give the best value per rupee at IPL auction, and which team is likely to win tonight?"*

This is the exact question Dream11's data team answers before every IPL auction and every match. This project builds the analytical system to answer it.

---

## 📊 Key Findings

| Insight | Finding |
|---|---|
| Toss impact | Winning toss and choosing to field wins **54%** of the time |
| Best death bowler | Lowest economy in overs 17–20 across all seasons |
| Highest auction value | Virat Kohli scores **92.4/100** on our custom metric |
| Six-hitting | Sixes per match have increased **3x** from 2008 to 2023 |
| Venue | Wankhede has the highest average first innings score |

---

## 🏗️ Project Architecture

```
ipl-analytics/
├── data/
│   ├── matches.csv          ← Raw Kaggle data
│   ├── deliveries.csv       ← 260k+ ball-by-ball records
│   └── ipl.db               ← SQLite database (auto-generated)
├── sql/
│   └── analysis.sql         ← 20+ advanced SQL queries
├── src/
│   ├── data_cleaning.py     ← Data pipeline & DB setup
│   ├── eda.py               ← 12 publication-ready charts
│   └── ml_model.py          ← Random Forest match predictor
├── visualizations/          ← All charts as PNG
├── app.py                   ← Streamlit dashboard (4 pages)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ipl-analytics.git
cd ipl-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the database (run once)
python src/data_cleaning.py

# 4. Generate all visualizations
python src/eda.py

# 5. Train the ML model
python src/ml_model.py

# 6. Launch the dashboard
streamlit run app.py
```

---

## 📋 Modules

### Module 1 — Data Pipeline
- Loads real Kaggle IPL data (1,095 matches, 260,920 deliveries)
- Cleans and structures into SQLite with optimised schema
- Creates pre-built views: `batting_summary`, `bowling_summary`
- Engineers features: phase of play, boundary flags, legal deliveries

### Module 2 — Advanced SQL (20+ queries)
Demonstrates senior-level SQL across 5 categories:

| Category | Techniques Used |
|---|---|
| Season overview | GROUP BY, aggregations |
| Batting analysis | RANK(), PARTITION BY, running totals |
| Bowling analysis | Economy rate, CTEs, subqueries |
| Team analysis | UNION, head-to-head, win % |
| Auction score | Custom composite metric formula |

### Module 3 — Python EDA (12 charts)
- Season-wise run scoring trend (dual axis)
- Top 10 batsmen and bowlers all-time
- Phase-wise run rate heatmap across seasons
- Toss decision impact analysis
- Death over specialist bowlers
- **Player Auction Value Score** — our original metric

### Module 4 — ML Model
- **Algorithm:** Random Forest Classifier (200 trees)
- **Features:** Teams, venue, toss winner, toss decision, season
- **Validation:** 5-fold cross-validation
- **Output:** Win probability for each team

### Module 5 — Streamlit Dashboard (4 pages)
| Page | What you can do |
|---|---|
| Season Overview | KPI cards, run trends, sixes explosion, title history |
| Player Analyser | Search any player — career stats, season breakdown, auction score |
| Team vs Team | Head-to-head record, match history, win distribution |
| Match Predictor | Select two teams and get win probability in real time |

---

## 💡 Original Metric — Player Auction Value Score

Most IPL projects just show who scored the most runs. We go further — a composite score that mimics how IPL auction analysts actually value players:

```
Auction Score =
    (Total Runs / 50)      × 30%   ← volume
  + (Strike Rate / 200)    × 25%   ← aggression  
  + (Batting Avg / 60)     × 20%   ← consistency
  + (Total Sixes / 20)     × 15%   ← six-hitting ability
  + (Player of Match / 5)  × 10%   ← match-winning impact
```

This produces a 0–100 score for every player with 15+ matches. Completely original — not available in any other IPL project on GitHub.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas | Data manipulation |
| SQLite + SQL | Database and advanced queries |
| Matplotlib / Seaborn | Static visualizations |
| Plotly | Interactive dashboard charts |
| Scikit-learn | Random Forest ML model |
| Streamlit | Interactive web dashboard |
| Joblib | Model serialization |

---

## 📁 Data Source

Real IPL ball-by-ball data from Kaggle:  
[IPL Complete Dataset 2008–2020](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

- `matches.csv` — 1,095 matches with toss, venue, winner details
- `deliveries.csv` — 260,920 ball-by-ball records

---

## 👤 Author

**Sridhar M**  
MBA Business Analytics | Data Analyst  
📧 sridharm6464@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

*Built as part of a data analytics portfolio. All analysis is based on publicly available data.*
