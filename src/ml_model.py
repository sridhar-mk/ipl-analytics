"""
IPL Analytics — Match Winner Predictor (Enhanced)
15 features: teams, venue, toss, historical win rate, head-to-head,
             venue advantage, recent form, win rate diff, form diff
"""

import sqlite3, os, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

BASE   = os.path.join(os.path.dirname(__file__), '..')
DB     = os.path.join(BASE, 'data', 'ipl.db')
OUTDIR = os.path.join(BASE, 'visualizations')
os.makedirs(OUTDIR, exist_ok=True)


def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("""
        SELECT id, season, team1, team2, city, venue,
               toss_winner, toss_decision, winner, bat_first
        FROM matches WHERE winner IS NOT NULL ORDER BY season, id
    """, conn)
    conn.close()
    return df


def compute_historical_features(df):
    df = df.copy().reset_index(drop=True)
    t1wr_list, t2wr_list, h2h_list, vr_list, t1rf_list, t2rf_list = [], [], [], [], [], []

    for i, row in df.iterrows():
        past = df.iloc[:i]

        def wr(team, data):
            played = data[(data['team1']==team)|(data['team2']==team)]
            return 0.5 if len(played)<3 else round((played['winner']==team).sum()/len(played),4)

        def h2h(t1, t2, data):
            m = data[((data['team1']==t1)&(data['team2']==t2))|((data['team1']==t2)&(data['team2']==t1))]
            return 0.5 if len(m)<2 else round((m['winner']==t1).sum()/len(m),4)

        def vrate(team, ven, data):
            m = data[(data['venue']==ven)&((data['team1']==team)|(data['team2']==team))]
            return 0.5 if len(m)<2 else round((m['winner']==team).sum()/len(m),4)

        def form(team, data, n=5):
            m = data[(data['team1']==team)|(data['team2']==team)].tail(n)
            return 0.5 if len(m)==0 else round((m['winner']==team).sum()/len(m),4)

        t1wr_list.append(wr(row['team1'], past))
        t2wr_list.append(wr(row['team2'], past))
        h2h_list.append(h2h(row['team1'], row['team2'], past))
        vr_list.append(vrate(row['team1'], row['venue'], past))
        t1rf_list.append(form(row['team1'], past))
        t2rf_list.append(form(row['team2'], past))

    df['team1_win_rate']    = t1wr_list
    df['team2_win_rate']    = t2wr_list
    df['h2h_team1_rate']    = h2h_list
    df['venue_team1_rate']  = vr_list
    df['team1_recent_form'] = t1rf_list
    df['team2_recent_form'] = t2rf_list
    df['win_rate_diff']     = df['team1_win_rate'] - df['team2_win_rate']
    df['form_diff']         = df['team1_recent_form'] - df['team2_recent_form']
    return df


def engineer_features(df):
    df = df.copy()
    df = compute_historical_features(df)
    df['team1_won']            = (df['winner']==df['team1']).astype(int)
    df['toss_winner_is_team1'] = (df['toss_winner']==df['team1']).astype(int)
    df['bat_first']            = (df['toss_decision']=='bat').astype(int)

    le = {}
    for col in ['team1','team2','venue','city']:
        le[col] = LabelEncoder()
        df[col+'_enc'] = le[col].fit_transform(df[col].fillna('Unknown'))

    features = [
        'team1_enc','team2_enc','venue_enc','city_enc',
        'toss_winner_is_team1','bat_first','season',
        'team1_win_rate','team2_win_rate','h2h_team1_rate',
        'venue_team1_rate','team1_recent_form','team2_recent_form',
        'win_rate_diff','form_diff'
    ]
    return df, features, le


def train_model(df, features):
    X = df[features]
    y = df['team1_won']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=4,
        random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Test Accuracy: {acc:.2%}  |  CV: {cv.mean():.2%} ± {cv.std():.2%}")
    print(classification_report(y_test, y_pred, target_names=['Team2','Team1']))
    return model, X_test, y_test, y_pred, acc


def save_model(model, le, features, df):
    history = df[['team1','team2','venue','city','winner','season','toss_decision']].copy()
    path = os.path.join(BASE, 'data', 'ipl_model.pkl')
    joblib.dump({'model':model,'encoders':le,'features':features,'history':history}, path)
    print(f"Model saved → {path}")
    return path


def predict_match(model_data, team1, team2, city, venue, toss_winner, toss_decision):
    model   = model_data['model']
    le      = model_data['encoders']
    history = model_data['history']

    def safe_enc(enc, val):
        return enc.transform([val])[0] if val in enc.classes_ else 0

    def wr(team, data):
        p = data[(data['team1']==team)|(data['team2']==team)]
        return 0.5 if len(p)<3 else round((p['winner']==team).sum()/len(p),4)

    def h2h(t1, t2, data):
        m = data[((data['team1']==t1)&(data['team2']==t2))|((data['team1']==t2)&(data['team2']==t1))]
        return 0.5 if len(m)<2 else round((m['winner']==t1).sum()/len(m),4)

    def vrate(team, ven, data):
        m = data[(data['venue']==ven)&((data['team1']==team)|(data['team2']==team))]
        return 0.5 if len(m)<2 else round((m['winner']==team).sum()/len(m),4)

    def form(team, data, n=5):
        m = data[(data['team1']==team)|(data['team2']==team)].tail(n)
        return 0.5 if len(m)==0 else round((m['winner']==team).sum()/len(m),4)

    t1wr = wr(team1, history);   t2wr = wr(team2, history)
    h2hr = h2h(team1, team2, history)
    vr   = vrate(team1, venue, history)
    t1rf = form(team1, history); t2rf = form(team2, history)

    X = pd.DataFrame([[
        safe_enc(le['team1'],team1), safe_enc(le['team2'],team2),
        safe_enc(le['venue'],venue), safe_enc(le['city'],city),
        1 if toss_winner==team1 else 0,
        1 if toss_decision=='bat' else 0,
        2024, t1wr, t2wr, h2hr, vr, t1rf, t2rf,
        t1wr-t2wr, t1rf-t2rf
    ]], columns=model_data['features'])

    proba = model.predict_proba(X)[0]
    breakdown = {
        'Overall win rate':     f"{team1}: {t1wr*100:.0f}%  |  {team2}: {t2wr*100:.0f}%",
        'Head to head':         f"{team1} wins {h2hr*100:.0f}% of past meetings",
        'Venue advantage':      f"{team1} wins {vr*100:.0f}% at this venue",
        'Recent form (last 5)': f"{team1}: {t1rf*100:.0f}%  |  {team2}: {t2rf*100:.0f}%",
        'Toss factor':          f"{'Batting' if toss_decision=='bat' else 'Fielding'} first",
    }
    return round(proba[1]*100,1), round(proba[0]*100,1), breakdown


if __name__ == '__main__':
    print("Training enhanced IPL predictor (15 features)...")
    df = load_data()
    df, features, le = engineer_features(df)
    model, *_ = train_model(df, features)
    save_model(model, le, features, df)