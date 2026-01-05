import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the temporal results
CSV_PATH = "data/groups/JC_Family/top_pairs_temporal.csv"

def main():
    if not Path(CSV_PATH).exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 1. Overall Error Metrics
    mae = df['abs_error'].mean()
    med_ae = df['abs_error'].median()
    total_games = df['games'].sum()
    weighted_mae = (df['abs_error'] * df['games']).sum() / total_games
    
    print(f"=== Overall Performance Metrics ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Median Absolute Error: {med_ae:.4f}")
    print(f"Weighted MAE (by games): {weighted_mae:.4f}")
    print(f"Total pairs analyzed: {len(df)}")
    print(f"Total matches in these pairs: {total_games}")

    # 2. Bias Analysis: real_wr - avg_pred_wr
    # If positive: Underestimated (player performed better than predicted)
    # If negative: Overestimated (player performed worse than predicted)
    df['bias'] = df['real_wr_p1'] - df['avg_pred_wr_p1']
    
    # 3. Correlation with Uncertainty (Sigma)
    df['avg_pair_sigma'] = (df['avg_sigma_p1'] + df['avg_sigma_p2']) / 2
    corr_sigma = df['abs_error'].corr(df['avg_pair_sigma'])
    
    print(f"\n=== Uncertainty Patterns ===")
    print(f"Correlation (Abs Error vs Avg Sigma): {corr_sigma:.4f}")
    # High correlation means the system 'knows' when it's going to be less accurate.

    # 4. Identifying Systemic Outliers (Players with most bias)
    # We need to map back to individual players
    player_stats = {} # name -> {'wins': 0, 'pred_wins': 0, 'games': 0}

    for _, row in df.iterrows():
        p1, p2 = row['p1_name'], row['p2_name']
        games = row['games']
        p1_real_wins = row['real_wr_p1'] * games
        p1_pred_wins = row['avg_pred_wr_p1'] * games
        
        for name in [p1, p2]:
            if name not in player_stats: player_stats[name] = {'wins': 0, 'pred_wins': 0, 'games': 0}
        
        player_stats[p1]['wins'] += p1_real_wins
        player_stats[p1]['pred_wins'] += p1_pred_wins
        player_stats[p1]['games'] += games
        
        player_stats[p2]['wins'] += (games - p1_real_wins)
        player_stats[p2]['pred_wins'] += (games - p1_pred_wins)
        player_stats[p2]['games'] += games

    player_bias_list = []
    from scipy.stats import binomtest
    
    for name, stats in player_stats.items():
        if stats['games'] >= 40: # Need enough games for significance
            real_wr = stats['wins'] / stats['games']
            pred_wr = stats['pred_wins'] / stats['games']
            bias = real_wr - pred_wr
            
            # P-value: Probability of seeing 'wins' or more extreme, given 'pred_wr'
            # Note: binomtest requires integer k, so we round
            k = int(round(stats['wins']))
            p_val = binomtest(k, int(stats['games']), pred_wr).pvalue
            
            player_bias_list.append({
                'name': name, 'bias': bias, 'p_val': p_val, 
                'games': int(stats['games']), 'real_wr': real_wr, 'pred_wr': pred_wr
            })

    bias_df = pd.DataFrame(player_bias_list)
    significant = bias_df[bias_df['p_val'] < 0.05].copy()
    
    print(f"\n=== Systemic Under-prediction (Actual > Predicted, p < 0.05) ===")
    print(significant[significant['bias'] > 0].sort_values('bias', ascending=False).head(15).to_string(index=False))
    
    print(f"\n=== Systemic Over-prediction (Actual < Predicted, p < 0.05) ===")
    print(significant[significant['bias'] < 0].sort_values('bias').head(15).to_string(index=False))

    # 5. Accuracy vs Predictive Confidence (Sharpness)
    # Does error increase at 50/50 predictions?
    df['certainty'] = abs(df['avg_pred_wr_p1'] - 0.5) * 2 # 0 at 50/50, 1 at 100/0
    corr_certainty = df['abs_error'].corr(df['certainty'])
    print(f"\n=== sharpness Patterns ===")
    print(f"Correlation (Abs Error vs Predictive Certainty): {corr_certainty:.4f}")

if __name__ == "__main__":
    main()
