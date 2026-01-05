import csv
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import trueskill
import numpy as np

# Configuration
WIN_PROB_TAU = 7.0
MATCHES_FILE = Path("data/matches_jc_filtered.csv")
K_CORE = 5

def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def compute_win_probability(mu_a: float, mu_b: float) -> float:
    return sigmoid((mu_a - mu_b) / WIN_PROB_TAU)

def load_matches():
    matches = []
    print("[*] Loading matches...")
    with open(MATCHES_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_status = int(row.get('p1_status', 0) or 0)
            p2_status = int(row.get('p2_status', 0) or 0)
            
            if p1_status == 1 and p2_status == 0:
                winner, loser = row['p1_id'], row['p2_id']
            elif p1_status == 0 and p2_status == 1:
                winner, loser = row['p2_id'], row['p1_id']
            else:
                continue
            
            # Parse time
            ts_str = row.get('start_time', '')
            try:
                # Get day-level datetime for batching
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                day = dt.date()
            except:
                continue
                
            matches.append({
                'winner': winner,
                'loser': loser,
                'time': dt,
                'day': day
            })
    
    # Sort chronologically
    matches.sort(key=lambda x: x['time'])
    print(f"    Loaded {len(matches)} matches.")
    return matches

def get_eligible_players(matches):
    import networkx as nx
    G = nx.Graph()
    for m in matches:
        G.add_edge(m['winner'], m['loser'])
    
    core = nx.k_core(G, k=K_CORE)
    return set(core.nodes())

def run_calibration(matches_by_day, eligible_players, drift_per_day):
    # Calibrate temporal skill drift (sigma growth per day) for the Bayesian Rating system.
    trueskill.setup(mu=25.0, sigma=25.0/3, beta=25.0/6, tau=0.0, draw_probability=0.0)
    
    ratings = {pid: trueskill.Rating() for pid in eligible_players}
    last_time = {} # pid -> datetime
    
    metrics = {
        'all': {'loss': 0, 'count': 0, 'correct': 0},
        'gap_0_7': {'loss': 0, 'count': 0, 'correct': 0},
        'gap_7_30': {'loss': 0, 'count': 0, 'correct': 0},
        'gap_30plus': {'loss': 0, 'count': 0, 'correct': 0}
    }
    
    days = sorted(matches_by_day.keys())
    warmup_days = len(days) // 10
    
    for i, day in enumerate(days):
        day_matches = matches_by_day[day]
        
        # 1. APPLY DRIFT to players in today's matches
        # (Based on when they last played before today)
        active_today = set()
        for m in day_matches:
            active_today.add(m['winner'])
            active_today.add(m['loser'])
            
        for pid in active_today:
            if pid in last_time:
                # Delta since their last match
                # Using the start of today's first match for this player roughly
                delta_days = (datetime.combine(day, datetime.min.time()) - 
                             datetime.combine(last_time[pid].date(), datetime.min.time())).days
                if delta_days > 0:
                    new_sigma = math.sqrt(ratings[pid].sigma**2 + drift_per_day * delta_days)
                    ratings[pid] = trueskill.Rating(mu=ratings[pid].mu, sigma=new_sigma)

        # 2. BATCH PREDICT today's matches (before updating)
        predictions = []
        for m in day_matches:
            p1, p2 = m['winner'], m['loser']
            if p1 not in eligible_players or p2 not in eligible_players:
                continue
                
            prob = compute_win_probability(ratings[p1].mu, ratings[p2].mu)
            
            # Determine max gap (how "stale" are the players?)
            gap1 = (day - last_time[p1].date()).days if p1 in last_time else 0
            gap2 = (day - last_time[p2].date()).days if p2 in last_time else 0
            max_gap = max(gap1, gap2)
            
            predictions.append({'prob': prob, 'max_gap': max_gap})
            
        # 3. UPDATE ratings with today's matches
        # (We process one by one but conceptually it's a day update)
        for m in day_matches:
            p1, p2 = m['winner'], m['loser']
            if p1 not in eligible_players or p2 not in eligible_players:
                continue
            
            ratings[p1], ratings[p2] = trueskill.rate_1vs1(ratings[p1], ratings[p2])
            last_time[p1] = m['time']
            last_time[p2] = m['time']

        # 4. STORE metrics for today
        if i >= warmup_days:
            for p in predictions:
                prob = min(max(p['prob'], 0.001), 0.999)
                loss = -math.log(prob)
                correct = 1 if prob > 0.5 else 0
                
                # All
                metrics['all']['loss'] += loss
                metrics['all']['count'] += 1
                metrics['all']['correct'] += correct
                
                # Segments
                label = 'gap_0_7'
                if p['max_gap'] > 30: label = 'gap_30plus'
                elif p['max_gap'] >= 7: label = 'gap_7_30'
                
                metrics[label]['loss'] += loss
                metrics[label]['count'] += 1
                metrics[label]['correct'] += correct
                
    return metrics

def main():
    all_matches = load_matches()
    eligible = get_eligible_players(all_matches)
    print(f"[*] Core players: {len(eligible)}")
    
    # Batch by day
    matches_by_day = defaultdict(list)
    for m in all_matches:
        matches_by_day[m['day']].append(m)
    
    drifts = [0, 0.005, 0.02, 0.1]
    
    print("\n[*] Starting DAY-BY-DAY calibration sweep...")
    header = f"{'Drift/Day':10} | {'Global Loss':12} | {'Acc %':8} | {'Gap 7-30':10} | {'Gap 30+'}"
    print(header)
    print("-" * len(header))
    
    for d in drifts:
        results = run_calibration(matches_by_day, eligible, d)
        
        all_l = results['all']['loss'] / results['all']['count']
        all_a = results['all']['correct'] / results['all']['count'] * 100
        
        g7_30 = results['gap_7_30']['loss'] / results['gap_7_30']['count'] if results['gap_7_30']['count'] > 0 else 0
        g30p = results['gap_30plus']['loss'] / results['gap_30plus']['count'] if results['gap_30plus']['count'] > 0 else 0
        
        print(f"{d:10.4f} | {all_l:12.6f} | {all_a:7.2f}% | {g7_30:10.6f} | {g30p:10.6f}")

if __name__ == "__main__":
    main()
