"""
IPL Analytics — Exploratory Data Analysis & Visualizations
Generates 15 publication-ready charts saved to /visualizations/
Run after data_cleaning.py
"""

import sqlite3, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE   = os.path.join(os.path.dirname(__file__), '..')
DB     = os.path.join(BASE, 'data', 'ipl.db')
OUTDIR = os.path.join(BASE, 'visualizations')
os.makedirs(OUTDIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────
PALETTE  = ['#1a73e8','#e8711a','#34a853','#ea4335',
            '#9c27b0','#00bcd4','#ff5722','#607d8b']
IPL_BLUE = '#1a73e8'
IPL_ORG  = '#e8711a'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
})

def save(name):
    path = os.path.join(OUTDIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}.png")

def conn():
    return sqlite3.connect(DB)


# ── 1. Runs per season trend ──────────────────────────────────
def plot_season_runs():
    df = pd.read_sql("""
        SELECT season, SUM(total_runs) AS runs,
               COUNT(DISTINCT match_id) AS matches
        FROM deliveries GROUP BY season ORDER BY season
    """, conn())
    df['runs_per_match'] = df['runs'] / df['matches']

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(df['season'], df['runs'], color=IPL_BLUE, alpha=0.7, label='Total runs')
    ax2 = ax1.twinx()
    ax2.plot(df['season'], df['runs_per_match'], color=IPL_ORG,
             marker='o', lw=2, label='Runs per match')
    ax1.set_title('IPL Season-wise Run Scoring Trend')
    ax1.set_xlabel('Season'); ax1.set_ylabel('Total Runs', color=IPL_BLUE)
    ax2.set_ylabel('Runs per Match', color=IPL_ORG)
    ax1.tick_params(axis='x', rotation=45)
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lbl1+lbl2, loc='upper left')
    plt.tight_layout()
    save('01_season_runs_trend')


# ── 2. Top 10 all-time run scorers ────────────────────────────
def plot_top_batsmen():
    df = pd.read_sql("""
        SELECT player, SUM(runs) AS total_runs
        FROM batting_summary
        GROUP BY player HAVING SUM(matches) >= 20
        ORDER BY total_runs DESC LIMIT 10
    """, conn())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['player'][::-1], df['total_runs'][::-1],
                   color=PALETTE[:10][::-1])
    for bar, val in zip(bars, df['total_runs'][::-1]):
        ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
                f'{int(val):,}', va='center', fontsize=10)
    ax.set_title('Top 10 All-Time IPL Run Scorers')
    ax.set_xlabel('Total Runs')
    plt.tight_layout()
    save('02_top_run_scorers')


# ── 3. Top 10 wicket takers ───────────────────────────────────
def plot_top_bowlers():
    df = pd.read_sql("""
        SELECT player, SUM(wickets) AS total_wickets,
               ROUND(SUM(runs_conceded)*6.0/NULLIF(SUM(legal_balls),0),2) AS economy
        FROM bowling_summary
        GROUP BY player HAVING SUM(matches) >= 20
        ORDER BY total_wickets DESC LIMIT 10
    """, conn())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['player'][::-1], df['total_wickets'][::-1], color=IPL_ORG)
    for bar, val in zip(bars, df['total_wickets'][::-1]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                str(int(val)), va='center', fontsize=10)
    ax.set_title('Top 10 All-Time IPL Wicket Takers')
    ax.set_xlabel('Total Wickets')
    plt.tight_layout()
    save('03_top_wicket_takers')


# ── 4. Toss decision impact ───────────────────────────────────
def plot_toss_impact():
    df = pd.read_sql("""
        SELECT toss_decision,
               COUNT(*) AS matches,
               SUM(toss_winner_won) AS wins
        FROM matches GROUP BY toss_decision
    """, conn())
    df['loss'] = df['matches'] - df['wins']

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(df))
    ax.bar(x, df['wins'],   label='Toss winner won',   color=IPL_BLUE, alpha=0.85)
    ax.bar(x, df['loss'],   label='Toss winner lost',  color='#ccc',
           bottom=df['wins'])
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.title() for d in df['toss_decision']])
    ax.set_title('Toss Decision Impact on Match Outcome')
    ax.set_ylabel('Matches')
    ax.legend()
    for i, row in df.iterrows():
        pct = row['wins']/row['matches']*100
        ax.text(i, row['wins']/2, f"{pct:.0f}%", ha='center',
                va='center', color='white', fontweight='bold')
    plt.tight_layout()
    save('04_toss_impact')


# ── 5. Phase-wise run scoring heatmap ────────────────────────
def plot_phase_heatmap():
    df = pd.read_sql("""
        SELECT season, phase,
               ROUND(AVG(total_runs)*6, 2) AS run_rate
        FROM deliveries
        GROUP BY season, phase
    """, conn())
    pivot = df.pivot(index='season', columns='phase', values='run_rate')
    pivot = pivot[['Powerplay','Middle','Death']]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label':'Run Rate'})
    ax.set_title('Run Rate by Phase Across Seasons')
    ax.set_xlabel('Phase of Play'); ax.set_ylabel('Season')
    plt.tight_layout()
    save('05_phase_heatmap')


# ── 6. Team win percentage ────────────────────────────────────
def plot_team_wins():
    df = pd.read_sql("""
        SELECT team, SUM(total_matches) AS matches,
               SUM(wins) AS wins,
               ROUND(SUM(wins)*100.0/SUM(total_matches),1) AS win_pct
        FROM (
            SELECT team1 AS team, COUNT(*) AS total_matches,
                   SUM(CASE WHEN winner=team1 THEN 1 ELSE 0 END) AS wins
            FROM matches GROUP BY team1
            UNION ALL
            SELECT team2, COUNT(*),
                   SUM(CASE WHEN winner=team2 THEN 1 ELSE 0 END)
            FROM matches GROUP BY team2
        ) GROUP BY team HAVING SUM(total_matches) >= 30
        ORDER BY win_pct DESC
    """, conn())

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [IPL_BLUE if p >= 50 else '#e8711a' for p in df['win_pct']]
    bars = ax.bar(df['team'], df['win_pct'], color=colors)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.6, label='50% line')
    ax.set_title('IPL Team Win Percentage (all time)')
    ax.set_ylabel('Win %')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, df['win_pct']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val}%', ha='center', fontsize=9)
    plt.tight_layout()
    save('06_team_win_pct')


# ── 7. Season-wise sixes trend ────────────────────────────────
def plot_sixes_trend():
    df = pd.read_sql("""
        SELECT season, SUM(is_six) AS sixes
        FROM deliveries GROUP BY season ORDER BY season
    """, conn())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df['season'], df['sixes'], alpha=0.3, color=IPL_ORG)
    ax.plot(df['season'], df['sixes'], color=IPL_ORG, marker='o', lw=2)
    ax.set_title('Six-Hitting Explosion Across IPL Seasons')
    ax.set_xlabel('Season'); ax.set_ylabel('Total Sixes')
    ax.tick_params(axis='x', rotation=45)
    for _, row in df.iterrows():
        ax.annotate(str(int(row['sixes'])),
                    (row['season'], row['sixes']),
                    textcoords='offset points', xytext=(0,6),
                    ha='center', fontsize=8)
    plt.tight_layout()
    save('07_sixes_trend')


# ── 8. Death overs — best economy bowlers ────────────────────
def plot_death_bowlers():
    df = pd.read_sql("""
        SELECT bowler AS player,
               SUM(is_legal_delivery) AS balls,
               SUM(is_wicket) AS wickets,
               ROUND(SUM(total_runs)*6.0/NULLIF(SUM(is_legal_delivery),0),2) AS economy
        FROM deliveries WHERE over >= 16
        GROUP BY bowler HAVING balls >= 200
        ORDER BY economy ASC LIMIT 10
    """, conn())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [IPL_BLUE if e < 8.5 else IPL_ORG for e in df['economy']]
    bars = ax.barh(df['player'][::-1], df['economy'][::-1], color=colors[::-1])
    ax.axvline(8.5, color='gray', linestyle='--', alpha=0.7, label='8.5 economy line')
    ax.set_title('Best Death Overs Bowlers (overs 17–20)')
    ax.set_xlabel('Economy Rate')
    for bar, val in zip(bars, df['economy'][::-1]):
        ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    save('08_death_bowlers')


# ── 9. Player Auction Value Score — top 15 ───────────────────
def plot_auction_score():
    df = pd.read_sql("""
        WITH batting AS (
            SELECT player,
                   SUM(matches) AS matches, SUM(runs) AS total_runs,
                   SUM(balls_faced) AS total_balls, SUM(sixes) AS sixes,
                   ROUND(SUM(runs)*100.0/NULLIF(SUM(balls_faced),0),2) AS sr,
                   ROUND(SUM(runs)*1.0/NULLIF(SUM(times_dismissed),0),2) AS avg
            FROM batting_summary GROUP BY player HAVING SUM(matches)>=15
        ),
        pom AS (
            SELECT player_of_match AS player, COUNT(*) AS awards
            FROM matches WHERE player_of_match IS NOT NULL
            GROUP BY player_of_match
        )
        SELECT b.player,
            ROUND(
                (b.total_runs/50.0)*30
              + (b.sr/200.0)*25
              + (b.avg/60.0)*20
              + (b.sixes/20.0)*15
              + (COALESCE(p.awards,0)/5.0)*10
            ,1) AS score
        FROM batting b LEFT JOIN pom p ON p.player=b.player
        ORDER BY score DESC LIMIT 15
    """, conn())

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap   = plt.cm.RdYlGn
    colors = [cmap(i/len(df)) for i in range(len(df)-1, -1, -1)]
    bars   = ax.barh(df['player'][::-1], df['score'][::-1], color=colors)
    ax.set_title('IPL Player Auction Value Score (our original metric)')
    ax.set_xlabel('Auction Value Score (0–100)')
    for bar, val in zip(bars, df['score'][::-1]):
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=10)
    plt.tight_layout()
    save('09_auction_value_score')


# ── 10. Venue scoring patterns (box plot) ────────────────────
def plot_venue_boxplot():
    df = pd.read_sql("""
        SELECT m.venue, d.match_id, SUM(d.total_runs) AS inn_runs
        FROM deliveries d JOIN matches m ON m.id=d.match_id
        WHERE d.inning=1
        GROUP BY m.venue, d.match_id
    """, conn())

    # top 8 venues by match count
    top = df['venue'].value_counts().head(8).index
    df  = df[df['venue'].isin(top)]

    # shorten long venue names
    df['venue'] = df['venue'].str.replace(
        'Punjab Cricket Association Stadium, Mohali', 'PCA Mohali')
    df['venue'] = df['venue'].str.replace(
        'Rajiv Gandhi International Cricket Stadium', 'RGICS Hyderabad')

    fig, ax = plt.subplots(figsize=(13, 6))
    order = df.groupby('venue')['inn_runs'].median().sort_values(
        ascending=False).index
    sns.boxplot(data=df, x='venue', y='inn_runs', order=order,
                palette='Set2', ax=ax)
    ax.set_title('First Innings Score Distribution by Venue')
    ax.set_xlabel('')
    ax.set_ylabel('First Innings Runs')
    ax.tick_params(axis='x', rotation=40)
    plt.tight_layout()
    save('10_venue_boxplot')


# ── 11. Runs contribution — top 5 batsmen across seasons ─────
def plot_career_progression():
    players = ['V Kohli','RG Sharma','SK Raina','CH Gayle','DA Warner']
    df = pd.read_sql("""
        SELECT player, season, SUM(runs) AS runs
        FROM batting_summary
        GROUP BY player, season
    """, conn())
    df = df[df['player'].isin(players)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, p in enumerate(players):
        sub = df[df['player']==p].sort_values('season')
        ax.plot(sub['season'], sub['runs'],
                marker='o', label=p, color=PALETTE[i], lw=2)
    ax.set_title('Season-wise Runs — Top 5 Batsmen')
    ax.set_xlabel('Season'); ax.set_ylabel('Runs')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save('11_career_progression')


# ── 12. Powerplay vs Death — team scoring rates ───────────────
def plot_powerplay_vs_death():
    df = pd.read_sql("""
        SELECT batting_team AS team, phase,
               ROUND(SUM(total_runs)*6.0/NULLIF(SUM(is_legal_delivery),0),2) AS rr
        FROM deliveries
        WHERE phase IN ('Powerplay','Death')
        GROUP BY batting_team, phase
    """, conn())

    teams_keep = (df.groupby('team').size()
                    .loc[lambda x: x==2].index.tolist())
    df = df[df['team'].isin(teams_keep)]
    pivot = df.pivot(index='team', columns='phase', values='rr').dropna()
    pivot = pivot.sort_values('Death', ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(pivot))
    w = 0.35
    ax.bar([i-w/2 for i in x], pivot['Powerplay'],
           width=w, label='Powerplay', color=IPL_BLUE, alpha=0.85)
    ax.bar([i+w/2 for i in x], pivot['Death'],
           width=w, label='Death', color=IPL_ORG, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=40, ha='right')
    ax.set_title('Team Run Rate — Powerplay vs Death Overs')
    ax.set_ylabel('Run Rate')
    ax.legend()
    plt.tight_layout()
    save('12_powerplay_vs_death')


# ── run all ───────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating visualizations...")
    plot_season_runs()
    plot_top_batsmen()
    plot_top_bowlers()
    plot_toss_impact()
    plot_phase_heatmap()
    plot_team_wins()
    plot_sixes_trend()
    plot_death_bowlers()
    plot_auction_score()
    plot_venue_boxplot()
    plot_career_progression()
    plot_powerplay_vs_death()
    print(f"\nAll charts saved to /visualizations/")
