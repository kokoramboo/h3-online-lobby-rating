import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the detailed temporal results
DETAILED_PATH = "data/groups/JC_Family/top_pairs_detailed.csv"

def main():
    if not Path(DETAILED_PATH).exists():
        print(f"Error: {DETAILED_PATH} not found.")
        return

    df = pd.read_csv(DETAILED_PATH)
    
    # 1. Error by Game Number
    # Group by game_num and calculate MAE
    learning = df.groupby('game_num').agg({
        'abs_error': 'mean',
        'p1_name': 'count'
    }).rename(columns={'p1_name': 'count'})
    
    print("=== Error Evolution by Game Number (Learning Curve) ===")
    # Look at buckets to keep it readable
    df['game_bucket'] = pd.cut(df['game_num'], 
                               bins=[0, 5, 10, 20, 50, 100, 500], 
                               labels=['1-5', '6-10', '11-20', '21-50', '51-100', '101+'])
    
    bucket_err = df.groupby('game_bucket', observed=True).agg({
        'abs_error': 'mean',
        'p1_name': 'count'
    }).rename(columns={'p1_name': 'count'})
    
    print(bucket_err.to_string())

    # 2. Uncertainty Impact
    # Calculate average sigma and error
    df['avg_sigma'] = (df['sigma1'] + df['sigma2']) / 2
    df['sigma_bucket'] = pd.cut(df['avg_sigma'], 
                                bins=[0, 1.0, 2.0, 4.0, 8.0], 
                                labels=['Stable (<1)', 'Mature (1-2)', 'Developing (2-4)', 'New (4+)'])
    
    sigma_err = df.groupby('sigma_bucket', observed=True).agg({
        'abs_error': 'mean',
        'p1_name': 'count'
    }).rename(columns={'p1_name': 'count'})
    
    print("\n=== Error by Uncertainty (Sigma) ===")
    print(sigma_err.to_string())

    print("\nSummary:")
    early_mae = df[df['game_num'] <= 10]['abs_error'].mean()
    late_mae = df[df['game_num'] > 10]['abs_error'].mean()
    print(f"MAE in First 10 Games: {early_mae:.4f}")
    print(f"MAE after 10 Games: {late_mae:.4f}")
    print(f"Improvement: {((early_mae - late_mae) / early_mae) * 100:.1f}%")

if __name__ == "__main__":
    main()
