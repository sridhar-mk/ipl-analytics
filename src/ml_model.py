"""
IPL Analytics — Match Winner Predictor
Random Forest classifier trained on 16 seasons of IPL data.
Features: teams, venue, toss decision, toss winner.
Run after data_cleaning.py
"""

import sqlite3, os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

BASE   = os.path.join(os.path.dirname(__file__), '..')
DB     = os.path.join(BASE, 'data', 'ipl.db')
OUTDIR = os.path.join(BASE, 'visualizations')
os.makedirs(OUTDIR, exist_ok=True)


def load_data():
    df = pd.read_sql("""
        SELECT team1, team2, city, toss_winner, toss_decision,
               winner, season
        FROM matches
        WHERE winner IS NOT NULL
    """, sqlite3.connect(DB))
    return df


def engineer_features(df):
    df = df.copy()

    # Target: did team1 win?
    df['team1_won'] = (df['winner'] == df['team1']).astype(int)

    # Toss features
    df['toss_winner_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    df['bat_first'] = (df['toss_decision'] == 'bat').astype(int)

    # Encode categorical columns
    le = {}
    for col in ['team1', 'team2', 'city']:
        le[col] = LabelEncoder()
        df[col + '_enc'] = le[col].fit_transform(df[col].fillna('Unknown'))

    features = ['team1_enc', 'team2_enc', 'city_enc',
                'toss_winner_is_team1', 'bat_first', 'season']
    return df, features, le


def train_model(df, features):
    X = df[features]
    y = df['team1_won']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print(f"\nModel Performance")
    print(f"{'─'*40}")
    print(f"Test Accuracy    : {acc:.2%}")
    print(f"CV Accuracy (5f) : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Team2 Wins','Team1 Wins']))

    return model, X_test, y_test, y_pred, acc


def plot_feature_importance(model, features):
    fi = pd.DataFrame({
        'feature': ['Team 1', 'Team 2', 'City / Venue',
                    'Toss Winner = Team1', 'Bat First', 'Season'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#1a73e8' if v >= fi['importance'].median()
              else '#ccc' for v in fi['importance']]
    ax.barh(fi['feature'], fi['importance'], color=colors)
    ax.set_title('Feature Importance — Match Winner Predictor')
    ax.set_xlabel('Importance Score')
    for i, (_, row) in enumerate(fi.iterrows()):
        ax.text(row['importance']+0.002, i,
                f"{row['importance']:.3f}", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '13_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 13_feature_importance.png")


def plot_confusion_matrix(y_test, y_pred):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Team2 Wins','Team1 Wins']
                           ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Match Predictor')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, '14_confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 14_confusion_matrix.png")


def save_model(model, le, features):
    model_path = os.path.join(BASE, 'data', 'ipl_model.pkl')
    joblib.dump({'model': model, 'encoders': le, 'features': features},
                model_path)
    print(f"\nModel saved to {model_path}")
    return model_path


def predict_match(model_data, team1, team2, city, toss_winner, toss_decision):
    """
    Standalone prediction function used by Streamlit app.
    Returns (win_probability_team1, win_probability_team2)
    """
    model = model_data['model']
    le    = model_data['encoders']

    def safe_encode(encoder, value):
        classes = list(encoder.classes_)
        if value not in classes:
            return 0
        return encoder.transform([value])[0]

    team1_enc  = safe_encode(le['team1'], team1)
    team2_enc  = safe_encode(le['team2'], team2)
    city_enc   = safe_encode(le['city'],  city)
    toss_t1    = 1 if toss_winner == team1 else 0
    bat        = 1 if toss_decision == 'bat' else 0

    X = pd.DataFrame([[team1_enc, team2_enc, city_enc, toss_t1, bat, 2023]],
                     columns=['team1_enc','team2_enc','city_enc',
                               'toss_winner_is_team1','bat_first','season'])
    proba = model.predict_proba(X)[0]
    return round(proba[1]*100, 1), round(proba[0]*100, 1)


if __name__ == '__main__':
    print("Training IPL Match Winner Predictor...")
    df              = load_data()
    df, features, le = engineer_features(df)
    model, Xt, yt, yp, acc = train_model(df, features)
    plot_feature_importance(model, features)
    plot_confusion_matrix(yt, yp)
    save_model(model, le, features)
    print(f"\nDone. Model accuracy: {acc:.2%}")
    print("Run app.py next for the Streamlit dashboard.")
