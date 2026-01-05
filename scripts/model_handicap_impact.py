import csv
import math
import json
import trueskill
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from scipy.optimize import minimize
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

def logit(p):
    return math.log(p / (1 - p))

def sigmoid(x):
    if x > 20: return 0.9999
    if x < -20: return 0.0001
    return 1.0 / (1.0 + math.exp(-x))

def compute_logit_trueskill(mu1, sigma1, mu2, sigma2, tau=DEFAULT_TAU):
    beta = tau * ((25.0/6.0)/5.50)
    delta_mu = mu1 - mu2
    sum_sigma_sq = sigma1**2 + sigma2**2 + 2 * (beta**2)
    s = math.sqrt(sum_sigma_sq)
    return 1.702 * delta_mu / s

def main():
    group_dir = settings.get_group_dir(GROUP)
    ratings_path = group_dir / "ratings.csv"
    matches_path = group_dir / "matches.csv"
    params_path = group_dir / "params.json"
    
    # 1. Identify Top 1000
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
    print(f"[*] Identified Top 1000 players.")

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
            if row.get('template') != "Jebus Cross": continue
            if row.get('is_random') != '0': continue
            
            t_str = row.get('start_time', '')
            try:
                dt = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
            except:
                dt = datetime.min.replace(tzinfo=timezone.utc)
            
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            if p1_s == 1 and p2_s == 0: outcome = 1.0
            elif p1_s == 0 and p2_s == 1: outcome = 0.0
            else: continue
            
            p1_h = float(row.get('p1_handicap', 0) or 0)
            p2_h = float(row.get('p2_handicap', 0) or 0)
            if abs(p1_h) > 10000 or abs(p2_h) > 10000: continue
            
            matches.append({
                'dt': dt, 'p1_id': row['p1_id'], 'p2_id': row['p2_id'], 
                'outcome': outcome,
                'p1_town': int(row['p1_town']), 'p2_town': int(row['p2_town']),
                'p1_color': int(row.get('p1_color', 0) or 0),
                'p1_handicap': p1_h,
                'p2_handicap': p2_h,
                'template': row.get('template'), 'map_size': row.get('map_size'),
                'is_random': row.get('is_random')
            })
    matches.sort(key=lambda x: x['dt'])
    print(f"[*] Loaded {len(matches)} elite-relevant matches.")

    # 4. Chronological Simulation to build dataset
    ratings = defaultdict(trueskill.Rating)
    last_time = {}
    
    data_points = []
    town_pairs = set()

    print("[*] Simulating skill and collecting handicap data...")
    total = len(matches)
    for i, m in enumerate(matches):
        if i % 100000 == 0: print(f"  {i}/{total} matches...")
        
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

        # Collect data for elite vs elite
        if p1 in top_1000 and p2 in top_1000:
            key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
            tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
            
            skill_logit = compute_logit_trueskill(ratings[p1].mu, ratings[p1].sigma, ratings[p2].mu, ratings[p2].sigma, tau)
            gold_diff = m['p1_handicap'] - m['p2_handicap']
            
            # Normalize gold to units of 1000 for stability
            gold_k = gold_diff / 1000.0
            
            # Triple = (p1_town, p2_town, p1_color)
            triplet = (m['p1_town'], m['p2_town'], m['p1_color'])
            town_pairs.add(triplet)
            
            data_points.append({
                'logit_skill': skill_logit,
                'gold_k': gold_k,
                'triplet': triplet,
                'outcome': m['outcome']
            })

        # Update Skill
        key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
        tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
        engine = trueskill.TrueSkill(mu=25.0, sigma=25.0/3.0, beta=tau * ((25.0/6.0)/5.50), tau=0, draw_probability=0.01)
        
        r1, r2 = ratings[p1], ratings[p2]
        if m['outcome'] == 1.0:
            ratings[p1], ratings[p2] = engine.rate_1vs1(r1, r2)
        else:
            ratings[p2], ratings[p1] = engine.rate_1vs1(r2, r1)

    if not data_points:
        print("No valid data points found.")
        return

    # Identify common triplets to reduce parameter count
    triplet_counts = defaultdict(int)
    for d in data_points: triplet_counts[d['triplet']] += 1
    
    # Threshold for triplet-specific modeling
    top_triplets = [p for p, c in triplet_counts.items() if c >= 200] 
    triplet_to_idx = {p: i for i, p in enumerate(top_triplets)}
    num_triplets = len(top_triplets)

    # 5. Prepare Vectorized Data
    logit_skill_arr = np.array([d['logit_skill'] for d in data_points])
    gold_k_arr = np.array([d['gold_k'] for d in data_points])
    outcomes_arr = np.array([d['outcome'] for d in data_points])
    trip_indices = np.array([triplet_to_idx.get(d['triplet'], -1) for d in data_points])
    
    print(f"[*] Data Summary:")
    print(f"    - Samples: {len(data_points)}")
    print(f"    - Triplets with 200+ games: {num_triplets}")
    print(f"    - Gold Range: {np.min(gold_k_arr):.1f}k to {np.max(gold_k_arr):.1f}k")
    print(f"    - Win Rate: {np.mean(outcomes_arr):.2%}")
    
    def loss(params):
        a, b = params[0], params[1] # Cubic, Quadratic (Global)
        c_global = params[2] 
        # Triplet specific params: bias and linear slope adjustment
        biases = params[3 : 3 + num_triplets]
        slopes_adj = params[3 + num_triplets : ]
        
        # Base Linear Slope = c_global + slopes_adj[idx]
        full_slopes = np.zeros(num_triplets + 1) + c_global
        full_slopes[:num_triplets] += slopes_adj
        slopes_selected = full_slopes[trip_indices]
        
        # Gold utility (3rd order hybrid)
        u_gold = a * (gold_k_arr**3) + b * (gold_k_arr**2) + slopes_selected * gold_k_arr
        
        # Biases
        full_biases = np.zeros(num_triplets + 1)
        full_biases[:num_triplets] = biases
        biases_selected = full_biases[trip_indices]
        
        logits = logit_skill_arr + u_gold + biases_selected
        
        # Stable Log-Likelihood
        ll = outcomes_arr * (-np.logaddexp(0, -logits)) + (1 - outcomes_arr) * (-np.logaddexp(0, logits))
        return -np.sum(ll)

    print(f"[*] Fitting 3rd order hybrid polynomial model for {num_triplets} triplets...")
    # Global: a, b, c; Local: num_triplets * (bias, slope_adj)
    initial_params = np.zeros(3 + 2 * num_triplets)
    initial_params[2] = 0.05 # Initial guess for global linear slope
    
    iter_count = 0
    def callback(xk):
        nonlocal iter_count
        iter_count += 1
        if iter_count % 10 == 0:
            print(f"  Optimization Iteration {iter_count}...")

    res = minimize(loss, initial_params, method='L-BFGS-B', options={'maxiter': 500}, callback=callback)
    
    a_fit, b_fit = res.x[0], res.x[1]
    c_global = res.x[2]
    biases_fit = res.x[3 : 3 + num_triplets]
    slopes_adj_fit = res.x[3 + num_triplets : ]

    print("\n" + "="*50)
    print("GOLD UTILITY MODEL (logit scale, G in 1000s)")
    print("="*50)
    print(f"Global Curves: a={a_fit:.6f}, b={b_fit:.6f}, c={c_global:.6f}")
    
    print("\n" + "="*50)
    print("FAIR PRICE HANDICAPS (WRAE=0, Equal Players)")
    print("="*50)
    print(f"{'Triplet Matchup':35} | {'Slope':8} | {'Fair Gold':10}")
    
    def solve_for_g(bias, slope):
        # Solve a*G^3 + b*G^2 + slope*G + bias = 0
        best_g = 0
        min_diff = 1e10
        # Check range -30k to 30k
        for test_g in np.linspace(-30, 30, 6001):
            val = a_fit*(test_g**3) + b_fit*(test_g**2) + slope*test_g + bias
            if abs(val) < min_diff:
                min_diff = abs(val)
                best_g = test_g
        return best_g * 1000.0

    table = []
    for tri, pidx in triplet_to_idx.items():
        t1, t2, c1 = tri
        name = f"{TOWN_MAP.get(t1)} vs {TOWN_MAP.get(t2)} ({'Red' if c1==1 else 'Blue'})"
        bias = biases_fit[pidx]
        slope = c_global + slopes_adj_fit[pidx]
        fair_g = solve_for_g(bias, slope)
        table.append((name, slope, fair_g))
    
    table.sort(key=lambda x: abs(x[2]), reverse=True)
    for name, s, g in table[:40]:
        print(f"{name:35} | {s:+.4f} | {g:+.0f} gold")

if __name__ == "__main__":
    main()
