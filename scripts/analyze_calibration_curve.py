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
    
    # We want to see: if we predict X%, what is the real Y%?
    # To get a full 0-1 range, we use P1's perspective
    # But to avoid p1/p2 arbitrary order, let's just use all data points
    # Actually, the CSV already gives us avg_pred_wr_p1 and real_wr_p1.
    
    # Bin the predictions
    bins = np.linspace(0, 1, 11) # 0, 0.1, ..., 1.0
    df['bin'] = pd.cut(df['avg_pred_wr_p1'], bins=bins, labels=bins[:-1] + 0.05)
    
    # Calculate real win rate per bin
    calibration = df.groupby('bin', observed=True).agg({
        'real_wr_p1': ['mean', 'count'],
        'avg_pred_wr_p1': 'mean'
    }).reset_index()
    
    calibration.columns = ['bin_center', 'real_mean', 'game_pairs', 'pred_mean']
    
    print("=== Reliability Diagram (Calibration Table) ===")
    print(calibration.to_string(index=False))
    
    print("\nInterpretation:")
    print("- If real_mean > pred_mean: System is too cautious (underestimating the better player).")
    print("- If real_mean < pred_mean: System is too aggressive (overestimating the better player).")
    
    # Overall bias in win probability
    # If we only look at bins > 0.5
    calibration['bin_center'] = calibration['bin_center'].astype(float)
    over_50 = calibration[calibration['bin_center'] > 0.5]
    if not over_50.empty:
        avg_under = (over_50['real_mean'] - over_50['pred_mean']).mean()
        print(f"\nAvg 'Caution' Bias (Bins > 0.5): {avg_under:.4f}")
        if avg_under > 0:
            print("System is systematically too cautious. Skill matters MORE than the current TAU suggests.")
        else:
            print("System is systematically too aggressive. Skill matters LESS than the current TAU suggests.")

if __name__ == "__main__":
    main()
