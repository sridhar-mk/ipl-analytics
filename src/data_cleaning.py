"""
IPL Analytics — Data Cleaning & Database Setup
Loads matches.csv and deliveries.csv into a structured SQLite database.
Run this first before anything else.
"""

import pandas as pd
import sqlite3
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_PATH  = os.path.join(DATA_DIR, 'ipl.db')


def load_raw():
    matches    = pd.read_csv(os.path.join(DATA_DIR, 'matches.csv'))
    deliveries = pd.read_csv(os.path.join(DATA_DIR, 'deliveries.csv'))
    print(f"Loaded {len(matches):,} matches and {len(deliveries):,} deliveries")
    return matches, deliveries


def clean_matches(df):
    # Normalise season label  e.g. "2007/08" → 2008
    def parse_season(s):
        s = str(s).strip()
        if '/' in s:
            return int(s.split('/')[0])  # 2007/08 → 2007, 2020/21 → 2020
        return int(s)

    df = df.copy()
    df['season'] = df['season'].apply(parse_season)

    # Fill missing cities from venue name (best effort)
    venue_city = {
        'M Chinnaswamy Stadium': 'Bangalore',
        'Wankhede Stadium': 'Mumbai',
        'Eden Gardens': 'Kolkata',
        'Feroz Shah Kotla': 'Delhi',
        'Arun Jaitley Stadium': 'Delhi',
        'MA Chidambaram Stadium': 'Chennai',
        'Rajiv Gandhi International Cricket Stadium': 'Hyderabad',
        'Sawai Mansingh Stadium': 'Jaipur',
        'Punjab Cricket Association Stadium, Mohali': 'Chandigarh',
        'Maharashtra Cricket Association Stadium': 'Pune',
        'Dubai International Cricket Stadium': 'Dubai',
        'Sheikh Zayed Stadium': 'Abu Dhabi',
        'Sharjah Cricket Stadium': 'Sharjah',
        'Narendra Modi Stadium': 'Ahmedabad',
    }
    df['city'] = df.apply(
        lambda r: venue_city.get(r['venue'], r['city'])
        if pd.isna(r['city']) else r['city'],
        axis=1
    )

    # Drop rows with no winner (abandoned matches)
    df = df.dropna(subset=['winner'])

    # Toss decision binary
    df['bat_first'] = (df['toss_decision'] == 'bat').astype(int)
    df['toss_winner_won'] = (df['toss_winner'] == df['winner']).astype(int)

    df = df.reset_index(drop=True)
    print(f"Cleaned matches: {len(df):,} rows")
    return df


def clean_deliveries(df, matches):
    df = df.copy()

    # Remove wide and no-ball extras from ball count for economy calc
    df['is_legal_delivery'] = (~df['extras_type'].isin(['wides', 'noballs'])).astype(int)

    # Merge season from matches
    season_map = matches.set_index('id')['season'].to_dict()
    df['season'] = df['match_id'].map(season_map)

    # Boundary flag
    df['is_boundary'] = df['batsman_runs'].isin([4, 6]).astype(int)
    df['is_six']      = (df['batsman_runs'] == 6).astype(int)
    df['is_four']     = (df['batsman_runs'] == 4).astype(int)

    # Phase of play
    def phase(over):
        if over < 6:   return 'Powerplay'
        if over < 15:  return 'Middle'
        return 'Death'

    df['phase'] = df['over'].apply(phase)

    print(f"Cleaned deliveries: {len(df):,} rows")
    return df


def build_database(matches, deliveries):
    conn = sqlite3.connect(DB_PATH)

    matches.to_sql('matches',    conn, if_exists='replace', index=False)
    deliveries.to_sql('deliveries', conn, if_exists='replace', index=False)

    # Pre-built batting summary view
    conn.execute("""
        CREATE VIEW IF NOT EXISTS batting_summary AS
        SELECT
            batter                                    AS player,
            season,
            COUNT(DISTINCT match_id)                  AS matches,
            SUM(batsman_runs)                         AS runs,
            COUNT(*)                                  AS balls_faced,
            SUM(is_boundary)                          AS boundaries,
            SUM(is_four)                              AS fours,
            SUM(is_six)                               AS sixes,
            ROUND(SUM(batsman_runs) * 100.0
                  / NULLIF(COUNT(*), 0), 2)           AS strike_rate,
            SUM(is_wicket)                            AS times_dismissed
        FROM deliveries
        GROUP BY batter, season
    """)

    # Pre-built bowling summary view
    conn.execute("""
        CREATE VIEW IF NOT EXISTS bowling_summary AS
        SELECT
            bowler                                          AS player,
            season,
            COUNT(DISTINCT match_id)                        AS matches,
            SUM(is_wicket)                                  AS wickets,
            SUM(total_runs)                                 AS runs_conceded,
            SUM(is_legal_delivery)                          AS legal_balls,
            ROUND(SUM(total_runs) * 6.0
                  / NULLIF(SUM(is_legal_delivery), 0), 2)  AS economy,
            ROUND(SUM(is_legal_delivery) * 1.0
                  / NULLIF(SUM(is_wicket), 0), 2)          AS bowling_average
        FROM deliveries
        GROUP BY bowler, season
    """)

    conn.commit()
    conn.close()
    print(f"Database built at: {DB_PATH}")


if __name__ == '__main__':
    matches, deliveries   = load_raw()
    matches               = clean_matches(matches)
    deliveries            = clean_deliveries(deliveries, matches)
    build_database(matches, deliveries)
    print("\nAll done. Run analysis.py next.")