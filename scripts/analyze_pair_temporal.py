import csv
import math
import json
import trueskill
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import settings

# Constants (must match rating_system.py)
SKILL_DRIFT_PER_DAY = 0.005
DEFAULT_TAU = 5.50
GROUP = "JC_Family"

def create_engine(tau):
    return trueskill.TrueSkill(
        mu=25.0, 
        sigma=25.0/3.0, 
        beta=tau * ((25.0/6.0)/5.50), 
        tau=0.0, 
        draw_probability=0.01
    )

def compute_prob_trueskill(mu1, sigma1, mu2, sigma2, tau=DEFAULT_TAU):
    """Note: beta in TS is tau * ratio"""
    beta = tau * ((25.0/6.0)/5.50)
    delta_mu = mu1 - mu2
    sum_sigma_sq = sigma1**2 + sigma2**2 + 2 * (beta**2)
    # Probit -> Logistic: Phi(x/s) approx sigmoid(1.702 * x / s)
    s = math.sqrt(sum_sigma_sq)
    return 1.0 / (1.0 + math.exp(-1.702 * delta_mu / s))

def main():
    group_dir = settings.get_group_dir(GROUP)
    ratings_path = group_dir / "ratings.csv"
    matches_path = group_dir / "matches.csv"
    params_path = group_dir / "params.json"
    players_path = settings.PLAYERS_CSV
    
    # 1. Load Player Names
    names = {}
    if players_path.exists():
        with open(players_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                names[row['player_id']] = row['nickname']

    # 2. Identify Top 1000
    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found")
        return
    
    top_1000 = []
    with open(ratings_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'eligible':
                top_1000.append(row['player_id'])
    # top_1000 is already sorted by norm_rating descending in ratings.csv
    top_1000 = set(top_1000[:1000])
    print(f"[*] Identified Top 1000 eligible players.")

    # 3. Load Params
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

    # 4. Prepare Matches
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
            if p1_s == 1 and p2_s == 0: winner, loser = row['p1_id'], row['p2_id']
            elif p1_s == 0 and p2_s == 1: winner, loser = row['p2_id'], row['p1_id']
            else: continue
            
            matches.append({
                'dt': dt, 'winner': winner, 'loser': loser, 'is_draw': p1_s == p2_s,
                'p1_id': row['p1_id'], 'p2_id': row['p2_id'],
                'template': row.get('template'), 'map_size': row.get('map_size'),
                'is_random': row.get('is_random')
            })
    matches.sort(key=lambda x: x['dt'])
    print(f"[*] Sorted {len(matches)} matches chronologically.")

    # 5. Simulate Skil Evolution and Predict
    ratings = defaultdict(trueskill.Rating)
    last_time = {}
    engines = {}
    pair_predictions = defaultdict(list) # (p_low, p_high) -> [(outcome, prob, sigma1, sigma2)]

    for m in matches:
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

        # Prediction Phase (Only for Top 1000 pairs)
        if p1 in top_1000 and p2 in top_1000:
            # Determine TAU
            key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
            tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
            
            # Predict
            prob = compute_prob_trueskill(ratings[p1].mu, ratings[p1].sigma, ratings[p2].mu, ratings[p2].sigma, tau)
            outcome = 1.0 if m['winner'] == p1 else 0.0
            
            pair = tuple(sorted([p1, p2]))
            game_idx = len(pair_predictions[pair]) + 1
            # Record from p1's perspective if p1 is the 'low' ID in the pair key, else from p2's
            if p1 == pair[0]:
                pair_predictions[pair].append((outcome, prob, ratings[p1].sigma, ratings[p2].sigma, game_idx))
            else:
                pair_predictions[pair].append((1.0 - outcome, 1.0 - prob, ratings[p2].sigma, ratings[p1].sigma, game_idx))

        # Update Phase
        key_triple = (m['template'], str(m['map_size']), str(m['is_random']))
        if key_triple not in engines:
            tau = param_lookup.get(key_triple, param_lookup.get(m['template'], group_default_tau))
            engines[key_triple] = create_engine(tau)
        
        engine = engines[key_triple]
        w_id, l_id = m['winner'], m['loser']
        ratings[w_id], ratings[l_id] = engine.rate_1vs1(ratings[w_id], ratings[l_id], drawn=m['is_draw'])

    # 6. Save Detailed Per-Match Records
    detailed_path = group_dir / "top_pairs_detailed.csv"
    with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["p1_name", "p2_name", "game_num", "outcome", "pred_wr", "sigma1", "sigma2", "abs_error"])
        
        for pair, data in pair_predictions.items():
            if len(data) < 20: continue
            
            p1_name = names.get(pair[0], pair[0])
            p2_name = names.get(pair[1], pair[1])
            
            for outcome, prob, s1, s2, gnum in data:
                writer.writerow([
                    p1_name, p2_name, gnum, 
                    f"{outcome:.0f}", f"{prob:.3f}", 
                    f"{s1:.2f}", f"{s2:.2f}",
                    f"{abs(outcome - prob):.3f}"
                ])

    # 7. Aggregate Summary
    output_path = group_dir / "top_pairs_temporal.csv"
    valid_pairs = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["p1_name", "p2_name", "games", "real_wr_p1", "avg_pred_wr_p1", "avg_sigma_p1", "avg_sigma_p2", "abs_error"])
        
        for pair, data in pair_predictions.items():
            if len(data) < 20: continue
            
            outcomes, probs, sig1, sig2, _ = zip(*data)
            real_wr = np.mean(outcomes)
            avg_pred = np.mean(probs)
            avg_s1 = np.mean(sig1)
            avg_s2 = np.mean(sig2)
            
            writer.writerow([
                names.get(pair[0], pair[0]), 
                names.get(pair[1], pair[1]), 
                len(data),
                f"{real_wr:.3f}", f"{avg_pred:.3f}", 
                f"{avg_s1:.2f}", f"{avg_s2:.2f}",
                f"{abs(real_wr - avg_pred):.3f}"
            ])
            valid_pairs += 1

    print(f"[*] Temporal analysis complete. Processed {valid_pairs} elite pairs.")
    print(f"[*] Detailed results: {detailed_path}")
    print(f"[*] Summary results: {output_path}")

if __name__ == "__main__":
    main()
