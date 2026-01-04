import csv
import math
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

# Configuration
WIN_PROB_TAU = 5.50
MATCHES_FILE = Path("data/matches_jc_filtered.csv")
RATINGS_FILE = Path("data/ratings_jc.csv")

def compute_win_probability(mu_a: float, mu_b: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-(mu_a - mu_b) / WIN_PROB_TAU))
    except OverflowError:
        return 0.0 if (mu_a - mu_b) / WIN_PROB_TAU < 0 else 1.0

def load_ratings(filepath: Path) -> dict:
    ratings = {}
    if not filepath.exists():
        return ratings
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('mu'): continue
            ratings[row['player_id']] = {'mu': float(row['mu'])}
    return ratings

def main():
    ratings = load_ratings(RATINGS_FILE)
    if not ratings:
        print(f"[!] No ratings found at {RATINGS_FILE}")
        return

    # template_metrics: {template_name: {'total_matches': 0, 'abs_error': 0, 'log_loss': 0}}
    template_data = defaultdict(lambda: {'matches': 0, 'total_abs_error': 0.0, 'total_log_loss': 0.0})

    print(f"[*] Analyzing {MATCHES_FILE}...")
    
    with open(MATCHES_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_id, p2_id = row['p1_id'], row['p2_id']
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            template = row.get('template', 'Unknown')
            
            if p1_s == 1 and p2_s == 0: actual_outcome = 1.0
            elif p1_s == 0 and p2_s == 1: actual_outcome = 0.0
            else: continue # Draw or invalid
            
            if p1_id not in ratings or p2_id not in ratings: continue
            
            mu1, mu2 = ratings[p1_id]['mu'], ratings[p2_id]['mu']
            prob = compute_win_probability(mu1, mu2)
            
            # Metrics
            abs_err = abs(actual_outcome - prob)
            eps = 1e-15
            log_loss = -(actual_outcome * math.log(max(prob, eps)) + (1.0 - actual_outcome) * math.log(max(1.0 - prob, eps)))
            
            template_data[template]['matches'] += 1
            template_data[template]['total_abs_error'] += abs_err
            template_data[template]['total_log_loss'] += log_loss

    # Sort templates by match count
    sorted_templates = sorted(template_data.items(), key=lambda x: x[1]['matches'], reverse=True)

    print("\n" + "="*95)
    print(f"{'Template Name':35} | {'Matches':8} | {'WMAE':8} | {'Log-Loss':8}")
    print("-" * 95)
    
    total_m = 0
    total_ae = 0
    total_ll = 0

    for name, stats in sorted_templates:
        m = stats['matches']
        if m < 100: continue # Skip low sample size for clarity
        
        wmae = stats['total_abs_error'] / m
        ll = stats['total_log_loss'] / m
        
        print(f"{name:35} | {m:8} | {wmae:.4f} | {ll:.4f}")
        
        total_m += m
        total_ae += stats['total_abs_error']
        total_ll += stats['total_log_loss']

    print("-" * 95)
    print(f"{'OVERALL (JC Family)':35} | {total_m:8} | {total_ae/total_m:.4f} | {total_ll/total_m:.4f}")
    print("="*95)
    
    print("\n[NOTE] Higher Log-Loss/WMAE implies higher 'randomness' or lower predictability.")

if __name__ == "__main__":
    main()
