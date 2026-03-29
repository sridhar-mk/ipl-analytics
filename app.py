"""
IPL Player Performance Analytics — Streamlit Dashboard
4 pages: Season Overview · Player Analyser · Team vs Team · Match Predictor

Run: streamlit run app.py
"""

import streamlit as st
import sqlite3, os, joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── config ────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))

DB_PATH  = os.path.join(BASE, 'data', 'ipl.db')
MDL_PATH = os.path.join(BASE, 'data', 'ipl_model.pkl')

# ── auto-build database and model if not present ──────────────
@st.cache_resource
def setup():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(MDL_PATH):
        os.remove(MDL_PATH)
    from src.data_cleaning import load_raw, clean_matches, clean_deliveries, build_database
    matches, deliveries   = load_raw()
    matches               = clean_matches(matches)
    deliveries            = clean_deliveries(deliveries, matches)
    build_database(matches, deliveries)
    from src.ml_model import load_data, engineer_features, train_model, save_model
    df               = load_data()
    df, features, le = engineer_features(df)
    model, *_        = train_model(df, features)
    save_model(model, le, features)

setup()

from src.ml_model import predict_match

st.set_page_config(
    page_title = "IPL Analytics",
    page_icon  = "🏏",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── helpers ───────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data
def query(sql):
    return pd.read_sql(sql, get_conn())

@st.cache_resource
def load_model():
    return joblib.load(MDL_PATH)

def kpi(col, label, value, delta=None):
    col.metric(label, value, delta)

TEAMS = [
    'Chennai Super Kings','Mumbai Indians','Kolkata Knight Riders',
    'Royal Challengers Bangalore','Sunrisers Hyderabad',
    'Delhi Capitals','Rajasthan Royals','Punjab Kings',
    'Gujarat Titans','Lucknow Super Giants',
]
CITIES = [
    'Mumbai','Chennai','Kolkata','Bangalore','Hyderabad',
    'Delhi','Jaipur','Chandigarh','Pune','Ahmedabad',
    'Dubai','Abu Dhabi','Sharjah',
]

# ── sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/Indian_Premier_League_Logo.svg/200px-Indian_Premier_League_Logo.svg.png",
                 width=120)
st.sidebar.title("IPL Analytics")
page = st.sidebar.radio("Navigate", [
    "🏆 Season Overview",
    "🏏 Player Analyser",
    "⚔️ Team vs Team",
    "🎯 Match Predictor",
])
st.sidebar.markdown("---")
st.sidebar.caption("Built by Sridhar M · Data: Kaggle IPL Dataset 2008–2023")


# ══════════════════════════════════════════
# PAGE 1 — SEASON OVERVIEW
# ══════════════════════════════════════════
if page == "🏆 Season Overview":
    st.title("🏆 IPL Season Overview")

    seasons = query("SELECT DISTINCT season FROM matches ORDER BY season")['season'].tolist()
    sel_season = st.selectbox("Select Season", ["All Seasons"] + seasons)

    if sel_season == "All Seasons":
        where_m = "1=1"
        where_d = "1=1"
    else:
        where_m = f"season = {sel_season}"
        where_d = f"season = {sel_season}"

    # KPIs
    kpis = query(f"""
        SELECT COUNT(DISTINCT id) AS matches,
               (SELECT COUNT(*) FROM deliveries WHERE {where_d}) AS deliveries,
               (SELECT SUM(total_runs) FROM deliveries WHERE {where_d}) AS runs,
               (SELECT SUM(is_six) FROM deliveries WHERE {where_d}) AS sixes
        FROM matches WHERE {where_m}
    """).iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Matches Played",  f"{int(kpis['matches']):,}")
    kpi(c2, "Total Runs",      f"{int(kpis['runs']):,}")
    kpi(c3, "Total Sixes",     f"{int(kpis['sixes']):,}")
    kpi(c4, "Run Rate",        f"{kpis['runs']/kpis['deliveries']*6:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Runs per Season")
        df = query("""
            SELECT season, SUM(total_runs) AS runs,
                   COUNT(DISTINCT match_id) AS matches
            FROM deliveries GROUP BY season ORDER BY season
        """)
        df = df.dropna(subset=['season'])
        df['season'] = df['season'].astype(int).astype(str)
        df['rpm'] = (df['runs'] / df['matches']).round(1)
        fig = px.bar(df, x='season', y='runs', color='rpm',
                     color_continuous_scale='Blues',
                     labels={'season':'Season','runs':'Total Runs','rpm':'Runs/Match'})
        fig.update_layout(showlegend=False, height=350,
                          xaxis=dict(type='category', tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Six-Hitting Trend")
        df = query("""
            SELECT season, SUM(is_six) AS sixes
            FROM deliveries GROUP BY season ORDER BY season
        """)
        df['season'] = df['season'].astype(str)
        fig = px.area(df, x='season', y='sixes',
                      color_discrete_sequence=['#e8711a'],
                      labels={'season':'Season','sixes':'Total Sixes'})
        fig.update_layout(height=350,
                          xaxis=dict(type='category', tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Title Winners")
        df = query("""
            SELECT winner AS team, COUNT(*) AS titles
            FROM matches WHERE match_type='Final' AND winner IS NOT NULL
            GROUP BY winner ORDER BY titles DESC
        """)
        fig = px.bar(df, x='titles', y='team', orientation='h',
                     color='titles', color_continuous_scale='RdYlGn')
        fig.update_layout(height=350, showlegend=False,
                          yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Toss Impact on Result")
        df = query("""
            SELECT toss_decision,
                   SUM(toss_winner_won) AS won,
                   COUNT(*) - SUM(toss_winner_won) AS lost
            FROM matches GROUP BY toss_decision
        """)
        fig = go.Figure()
        fig.add_bar(name='Won', x=df['toss_decision'], y=df['won'],
                    marker_color='#1a73e8')
        fig.add_bar(name='Lost', x=df['toss_decision'], y=df['lost'],
                    marker_color='#e8711a')
        fig.update_layout(barmode='stack', height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 2 — PLAYER ANALYSER
# ══════════════════════════════════════════
elif page == "🏏 Player Analyser":
    st.title("🏏 Player Analyser")

    players = query("""
        SELECT DISTINCT player FROM batting_summary
        ORDER BY player
    """)['player'].tolist()

    player = st.selectbox("Search Player", players, index=players.index('V Kohli')
                          if 'V Kohli' in players else 0)

    batting = query(f"""
        SELECT season, matches, runs, balls_faced,
               strike_rate, times_dismissed,
               ROUND(runs*1.0/NULLIF(times_dismissed,0),2) AS avg,
               sixes, fours
        FROM batting_summary WHERE player='{player}'
        ORDER BY season
    """)

    bowling = query(f"""
        SELECT season, matches, wickets, runs_conceded,
               economy, bowling_average
        FROM bowling_summary WHERE player='{player}'
        ORDER BY season
    """)

    if batting.empty and bowling.empty:
        st.warning("No data found for this player.")
    else:
        # Career summary KPIs
        if not batting.empty:
            st.subheader(f"Batting — {player}")
            c1,c2,c3,c4,c5 = st.columns(5)
            kpi(c1, "Total Runs",    f"{int(batting['runs'].sum()):,}")
            kpi(c2, "Career Avg",    f"{batting['runs'].sum()/max(batting['times_dismissed'].sum(),1):.1f}")
            kpi(c3, "Career SR",     f"{batting['runs'].sum()*100/max(batting['balls_faced'].sum(),1):.1f}")
            kpi(c4, "Total Sixes",   f"{int(batting['sixes'].sum())}")
            kpi(c5, "Seasons",       str(len(batting)))

            fig = go.Figure()
            fig.add_bar(name='Runs', x=batting['season'],
                        y=batting['runs'], marker_color='#1a73e8')
            fig.add_scatter(name='Strike Rate', x=batting['season'],
                            y=batting['strike_rate'], yaxis='y2',
                            mode='lines+markers', marker_color='#e8711a')
            fig.update_layout(
                title=f"{player} — Season-wise Batting",
                yaxis=dict(title='Runs'),
                yaxis2=dict(title='Strike Rate', overlaying='y', side='right'),
                height=380, legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        if not bowling.empty and bowling['wickets'].sum() > 0:
            st.subheader(f"Bowling — {player}")
            c1,c2,c3 = st.columns(3)
            kpi(c1, "Total Wickets", str(int(bowling['wickets'].sum())))
            kpi(c2, "Career Economy",
                f"{bowling['runs_conceded'].sum()*6/max(1, bowling['wickets'].sum()*6):.2f}")
            kpi(c3, "Seasons Bowled", str(len(bowling)))

            fig = px.bar(bowling, x='season', y='wickets',
                         color='economy', color_continuous_scale='RdYlGn_r',
                         title=f"{player} — Wickets by Season")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Auction Value Score
        score_df = query(f"""
            WITH b AS (
                SELECT player,
                       SUM(matches) AS m, SUM(runs) AS r,
                       SUM(balls_faced) AS bf, SUM(sixes) AS s,
                       ROUND(SUM(runs)*100.0/NULLIF(SUM(balls_faced),0),2) AS sr,
                       ROUND(SUM(runs)*1.0/NULLIF(SUM(times_dismissed),0),2) AS avg
                FROM batting_summary WHERE player='{player}' GROUP BY player
            ),
            p AS (
                SELECT player_of_match AS player, COUNT(*) AS awards
                FROM matches WHERE player_of_match='{player}' GROUP BY player_of_match
            )
            SELECT ROUND(
                (b.r/50.0)*30 + (b.sr/200.0)*25 + (b.avg/60.0)*20
                + (b.s/20.0)*15 + (COALESCE(p.awards,0)/5.0)*10
            ,1) AS score
            FROM b LEFT JOIN p ON p.player=b.player
        """)

        if not score_df.empty and score_df['score'].iloc[0] is not None:
            score = float(score_df['score'].iloc[0])
            st.markdown("---")
            st.subheader("Auction Value Score")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"{player} — Auction Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1a73e8"},
                    'steps': [
                        {'range': [0,40],  'color': '#ffebee'},
                        {'range': [40,70], 'color': '#fff8e1'},
                        {'range': [70,100],'color': '#e8f5e9'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75, 'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════
# PAGE 3 — TEAM VS TEAM
# ══════════════════════════════════════════
elif page == "⚔️ Team vs Team":
    st.title("⚔️ Team vs Team")

    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Team 1", TEAMS, index=0)
    t2 = c2.selectbox("Team 2", TEAMS, index=1)

    if t1 == t2:
        st.warning("Please select two different teams.")
    else:
        h2h = query(f"""
            SELECT winner,
                   COUNT(*) AS matches,
                   AVG(result_margin) AS avg_margin
            FROM matches
            WHERE (team1='{t1}' AND team2='{t2}')
               OR (team1='{t2}' AND team2='{t1}')
            GROUP BY winner
        """)

        all_matches = query(f"""
            SELECT id, season, city, venue, winner, result, result_margin,
                   toss_winner, toss_decision, player_of_match
            FROM matches
            WHERE (team1='{t1}' AND team2='{t2}')
               OR (team1='{t2}' AND team2='{t1}')
            ORDER BY season DESC
        """)

        if all_matches.empty:
            st.info("These two teams haven't played each other yet.")
        else:
            t1_wins = int(h2h[h2h['winner']==t1]['matches'].sum()) if t1 in h2h['winner'].values else 0
            t2_wins = int(h2h[h2h['winner']==t2]['matches'].sum()) if t2 in h2h['winner'].values else 0
            total   = len(all_matches)

            col1,col2,col3 = st.columns(3)
            kpi(col1, f"{t1} Wins", str(t1_wins))
            kpi(col2, "Total Matches", str(total))
            kpi(col3, f"{t2} Wins", str(t2_wins))

            fig = go.Figure(go.Pie(
                labels=[t1, t2],
                values=[t1_wins, t2_wins],
                hole=0.5,
                marker_colors=['#1a73e8','#e8711a'],
            ))
            fig.update_layout(title=f"Head to Head — {t1} vs {t2}", height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Match History")
            st.dataframe(all_matches[['season','city','winner',
                                       'result','result_margin',
                                       'player_of_match']],
                         use_container_width=True, height=350)


# ══════════════════════════════════════════
# PAGE 4 — MATCH PREDICTOR
# ══════════════════════════════════════════
elif page == "🎯 Match Predictor":
    st.title("🎯 Match Winner Predictor")
    st.caption("Random Forest model trained on 16 seasons of IPL data")

    c1, c2 = st.columns(2)
    team1  = c1.selectbox("Team 1 (Home)", TEAMS, index=0)
    team2  = c2.selectbox("Team 2 (Away)", TEAMS, index=1)
    city   = st.selectbox("Venue City", CITIES)

    c3, c4 = st.columns(2)
    toss_winner  = c3.selectbox("Toss Winner", [team1, team2])
    toss_decision = c4.selectbox("Toss Decision", ["bat", "field"])

    if st.button("🔮 Predict Winner", type="primary"):
        if team1 == team2:
            st.error("Please select two different teams.")
        else:
            try:
                model_data = load_model()
                p1, p2 = predict_match(
                    model_data, team1, team2,
                    city, toss_winner, toss_decision
                )
                st.markdown("---")
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)

                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=p1,
                        number={'suffix': '%'},
                        title={'text': f"{team1} Win Probability"},
                        gauge={
                            'axis': {'range': [0,100]},
                            'bar': {'color': '#1a73e8'},
                            'steps': [
                                {'range':[0,40],'color':'#ffebee'},
                                {'range':[40,60],'color':'#fff8e1'},
                                {'range':[60,100],'color':'#e8f5e9'},
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=p2,
                        number={'suffix': '%'},
                        title={'text': f"{team2} Win Probability"},
                        gauge={
                            'axis': {'range': [0,100]},
                            'bar': {'color': '#e8711a'},
                            'steps': [
                                {'range':[0,40],'color':'#ffebee'},
                                {'range':[40,60],'color':'#fff8e1'},
                                {'range':[60,100],'color':'#e8f5e9'},
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                winner = team1 if p1 > p2 else team2
                prob   = max(p1, p2)
                st.success(f"Model predicts **{winner}** wins with **{prob}%** probability")
                st.info("Note: Predictions are based on historical patterns. Cricket is gloriously unpredictable!")

            except Exception as e:
                st.error(f"Run ml_model.py first to train the model. ({e})")