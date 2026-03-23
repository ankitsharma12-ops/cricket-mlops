import pandas as pd
import os

def load_data(matches_path, deliveries_path):
    """Load raw IPL data"""
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    print(f"✅ Matches loaded: {matches.shape}")
    print(f"✅ Deliveries loaded: {deliveries.shape}")
    return matches, deliveries

def prepare_features(matches):
    """Engineer features from matches data"""

    # Select relevant columns
    df = matches[[
        'team1', 'team2', 'venue',
        'toss_winner', 'toss_decision', 'winner'
    ]].copy()

    # Drop rows with missing winner (no result matches)
    df = df.dropna(subset=['winner'])

    # Target: did toss winner win the match?
    df['toss_win_match_win'] = (
        df['toss_winner'] == df['winner']
    ).astype(int)

    # Drop winner column — not needed as feature
    
    df = df.drop(columns=['winner'])
    # Encode categorical columns
    categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    print(f"✅ Features prepared: {df.shape}")
    print(f"✅ Target distribution:\n{df['toss_win_match_win'].value_counts()}")

    return df

def save_processed_data(df, output_path):
    """Save processed features to CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")

if __name__ == "__main__":
    # Paths
    MATCHES_PATH    = "data/raw/matches.csv"
    DELIVERIES_PATH = "data/raw/deliveries.csv"
    OUTPUT_PATH     = "data/processed/features.csv"

    # Run pipeline
    matches, deliveries = load_data(MATCHES_PATH, DELIVERIES_PATH)
    df = prepare_features(matches)
    save_processed_data(df, OUTPUT_PATH)