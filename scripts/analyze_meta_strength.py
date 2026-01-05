import csv
import math
import json
import trueskill
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import settings

# Constants
SKILL_DRIFT_PER_DAY = 0.005
DEFAULT_TAU = 5.25 # Final calibrated value for JC_Family
GROUP = "JC_Family"

TOWN_MAP = {
    0: "Castle",
    7: "Stronghold",
    4: "Necropolis",
    2: "Tower",
    5: "Dungeon",
    1: "Rampart",
    9: "Cove",
    3: "Inferno",
    6: "Fortress",
    8: "Conflux",
    10: "Factory"
}

def compute_prob_trueskill(mu1, sigma1, mu2, sigma2, tau=DEFAULT_TAU):
    beta = tau * ((25.0/6.0)/5.50)
    delta_mu = mu1 - mu2
    sum_sigma_sq = sigma1**2 + sigma2**2 + 2 * (beta**2)
    s = math.sqrt(sum_sigma_sq)
    return 1.0 / (1.0 + math.exp(-1.702 * delta_mu / s))

def main():
    group_dir = settings.get_group_dir(GROUP)
    ratings_path = group_dir / "ratings.csv"
    matches_path = group_dir / "matches.csv"
    params_path = group_dir / "params.json"
    
    # 1. Identify Top 1000 (Elite Pool)
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found")
        return
    
    top_1000 = []
    with open(ratings_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'eligible':
                top_1000.append(row['player_id'])
    top_1000 = set(top_1000[:1000])
    print(f"[*] Identified Top 1000 elite players.")

    # 2. Load Params
    param_lookup = {}
    group_default_tau = DEFAULT_TAU
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
            for p in params:
                if p.get("group_default"): group_default_tau = p["tau"]
                elif "template" in p:
                    if "map_size" in p and "is_random" in p:
                        key = (p["template"], str(p["map_size"]), str(p["is_random"]))
                        param_lookup[key] = p["tau"]
                    else: param_lookup[p["template"]] = p["tau"]

    # 3. Load and Sort Matches
    matches = []
    with open(matches_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_str = row.get('start_time', '')
            try:
                dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
            except:
                dt = datetime.min.replace(tzinfo=timezone.utc)
            
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            if p1_s == 1 and p2_s == 0: winner_id = row['p1_id']
            elif p1_s == 0 and p2_s == 1: winner_id = row['p2_id']
            else: continue # Skip draws/invalid for meta analysis
            
            matches.append({
                'dt': dt, 'p1_id': row['p1_id'], 'p2_id': row['p2_id'], 
                'winner_id': winner_id,
                'p1_color': int(row['p1_color']), 'p1_town': int(row['p1_town']),
                'p2_color': int(row['p2_color']), 'p2_town': int(row['p2_town']),
                'template': row.get('template'), 'map_size': row.get('map_size'),
                'is_random': row.get('is_random')
            })
    matches.sort(key=lambda x: x['dt'])
    print(f"[*] Sorted {len(matches)} matches.")

    # 4. Simulation and WRAE stats
    ratings = defaultdict(trueskill.Rating)
    last_time = {}
    
    # Stats containers
    side_stats = defaultdict(lambda: {"wrae": 0, "count": 0})
    town_stats = defaultdict(lambda: {"wrae": 0, "count": 0})
    engine_default = trueskill.TrueSkill(mu=25.0, sigma=25.0/3.0, beta=group_default_tau * ((25.0/6.0)/5.50), tau=0, draw_probability=0.01)
    
    matchup_stats = defaultdict(lambda: {"wrae": 0, "count": 0}) # (t1, t2) -> wrae from t1 perspective
    
    total = len(matches)
    for i, m in enumerate(matches):
        if i % 100000 == 0:
            print(f"  Progress: {i}/{total} matches...")
            
        p1, p2 = m['p1_id'], m['p2_id']
        t = m['dt']
        
        # Apply Drift
        for pid in [p1, p2]:
            if pid in last_time and t > datetime.min.replace(tzinfo=timezone.utc):
                delta_days = (t - last_time[pid]).total_seconds() / 86400.0
                if delta_days > 0:
                    new_sigma = math.sqrt(ratings[pid].sigma**2 + SKILL_DRIFT_PER_DAY * delta_days)
                    ratings[pid] = trueskill.Rating(mu=ratings[pid].mu, sigma=new_sigma)
            last_time[pid] = t

        # Core Analysis (Elite only + Strictly Jebus Cross + Non-Random Only)
        if p1 in top_1000 and p2 in top_1000 and m['template'] == "Jebus Cross" and m['is_random'] == '0':
            key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
            tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
            
            # Probability P1 wins based on skill
            prob1 = compute_prob_trueskill(ratings[p1].mu, ratings[p1].sigma, ratings[p2].mu, ratings[p2].sigma, tau)
            outcome1 = 1.0 if m['winner_id'] == p1 else 0.0
            
            wrae1 = outcome1 - prob1
            wrae2 = -wrae1
            
            # Record side advantage (strictly 0/1)
            for pid, pcolor, pwrae in [(p1, m['p1_color'], wrae1), (p2, m['p2_color'], wrae2)]:
                if pcolor in [0, 1]:
                    side_stats[pcolor]['wrae'] += pwrae
                    side_stats[pcolor]['count'] += 1
            
            # Record town advantage
            town_stats[m['p1_town']]['wrae'] += wrae1
            town_stats[m['p1_town']]['count'] += 1
            town_stats[m['p2_town']]['wrae'] += wrae2
            town_stats[m['p2_town']]['count'] += 1
            
            # Matchup matrix
            pair = (m['p1_town'], m['p2_town'])
            matchup_stats[pair]['wrae'] += wrae1
            matchup_stats[pair]['count'] += 1
            inv_pair = (m['p2_town'], m['p1_town'])
            matchup_stats[inv_pair]['wrae'] += wrae2
            matchup_stats[inv_pair]['count'] += 1

        # Update Skill
        key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
        tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
        engine = trueskill.TrueSkill(mu=25.0, sigma=25.0/3.0, beta=tau * ((25.0/6.0)/5.50), tau=0, draw_probability=0.01)
        
        if m['winner_id'] == p1:
            ratings[p1], ratings[p2] = engine.rate_1vs1(ratings[p1], ratings[p2])
        else:
            ratings[p2], ratings[p1] = engine.rate_1vs1(ratings[p2], ratings[p1])

    # 5. Report Results
    print("\n" + "="*40)
    print("SIDE ADVANTAGE (WRAE)")
    print("="*40)
    for color, stats in side_stats.items():
        name = "Red" if color == 1 else "Blue"
        avg_wrae = stats['wrae'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{name:10} | WRAE: {avg_wrae:+.4f} | Games: {stats['count']}")

    print("\n" + "="*40)
    print("TOWN STRENGTH (WRAE)")
    print("="*40)
    town_results = []
    for tid, stats in town_stats.items():
        name = TOWN_MAP.get(tid, f"Unknown ({tid})")
        avg_wrae = stats['wrae'] / stats['count'] if stats['count'] > 0 else 0
        town_results.append((name, avg_wrae, stats['count']))
    
    town_results.sort(key=lambda x: x[1], reverse=True)
    for name, wrae, count in town_results:
        print(f"{name:15} | WRAE: {wrae:+.4f} | Games: {count}")

    print("\n" + "="*40)
    print("TOP 20 TOWN MATCHUPS (WRAE)")
    print("="*40)
    matchup_results = []
    for pair, stats in matchup_stats.items():
        if stats['count'] >= 50: # Only significant samples
            t1, t2 = pair
            n1 = TOWN_MAP.get(t1, f"Unknown ({t1})")
            n2 = TOWN_MAP.get(t2, f"Unknown ({t2})")
            avg_wrae = stats['wrae'] / stats['count']
            matchup_results.append((n1, n2, avg_wrae, stats['count']))
    
    matchup_results.sort(key=lambda x: x[2], reverse=True)
    for n1, n2, wrae, count in matchup_results[:20]:
        print(f"{n1:15} vs {n2:15} | WRAE: {wrae:+.4f} | Games: {count}")

if __name__ == "__main__":
    main()
