import csv
import math
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

# Configuration
WIN_PROB_TAU = 5.50
MATCHES_FILE = Path("data/matches.csv")
# Usually ratings come from the main family (JC_Family)
RATINGS_FILE = Path("data/groups/JC_Family/ratings.csv")

def compute_win_probability(mu_a: float, mu_b: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-(mu_a - mu_b) / WIN_PROB_TAU))
    except (OverflowError, ZeroDivisionError):
        return 0.5

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
        print(f"[!] No ratings found at {RATINGS_FILE}. Predictability metrics will be skipped.")

    # template_metrics: {template_name: {'matches': 0, 'total_abs_error': 0.0, 'total_log_loss': 0.0}}
    template_data = defaultdict(lambda: {'matches': 0, 'total_abs_error': 0.0, 'total_log_loss': 0.0})

    print(f"[*] Analyzing {MATCHES_FILE}...")
    
    if not MATCHES_FILE.exists():
        print(f"[!] Matches file not found at {MATCHES_FILE}")
        return

    with open(MATCHES_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            template = row.get('template', 'Unknown')
            template_data[template]['matches'] += 1
            
            if not ratings:
                continue

            p1_id, p2_id = row['p1_id'], row['p2_id']
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            
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
            
            template_data[template]['total_abs_error'] += abs_err
            template_data[template]['total_log_loss'] += log_loss

    # Sort templates by match count
    sorted_templates = sorted(template_data.items(), key=lambda x: x[1]['matches'], reverse=True)

    # Save to CSV
    output_csv = Path("data/template_stats.csv")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['template', 'matches', 'wmae', 'log_loss'] if ratings else ['template', 'matches']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in sorted_templates:
            row = {'template': name, 'matches': stats['matches']}
            if ratings:
                m = stats['matches']
                row['wmae'] = stats['total_abs_error'] / m if m > 0 else 0
                row['log_loss'] = stats['total_log_loss'] / m if m > 0 else 0
            writer.writerow(row)
    print(f"[*] Results saved to {output_csv}")

    print("\n" + "="*115)
    if ratings:
        print(f"{'Template Name':45} | {'Matches':10} | {'WMAE':10} | {'Log-Loss':10}")
        print("-" * 115)
    else:
        print(f"{'Template Name':45} | {'Matches':10}")
        print("-" * 60)
    
    total_m = 0
    total_ae = 0
    total_ll = 0

    for name, stats in sorted_templates:
        m = stats['matches']
        if m < 10: continue # Very low threshold just to show counts
        
        if ratings:
            wmae = stats['total_abs_error'] / m if stats['total_abs_error'] > 0 else 0
            ll = stats['total_log_loss'] / m if stats['total_log_loss'] > 0 else 0
            print(f"{name:45} | {m:10} | {wmae:10.4f} | {ll:10.4f}")
            
            total_ae += stats['total_abs_error']
            total_ll += stats['total_log_loss']
        else:
            print(f"{name:45} | {m:10}")
            
        total_m += m

    print("-" * (115 if ratings else 60))
    if ratings and total_m > 0:
        print(f"{'OVERALL':45} | {total_m:10} | {total_ae/total_m:10.4f} | {total_ll/total_m:10.4f}")
    else:
        print(f"{'OVERALL':45} | {total_m:10}")
    print("=" * (115 if ratings else 60))
    
    if ratings:
        print("\n[NOTE] Higher Log-Loss/WMAE implies higher 'randomness' or lower predictability.")

if __name__ == "__main__":
    main()
