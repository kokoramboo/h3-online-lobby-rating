import csv
import math
from pathlib import Path
import numpy as np
import sys
import json
import settings

# Configuration (must match rating_system.py)
DEFAULT_TAU = 5.50

def load_param_lookup(group_dir: Path):
    param_lookup = {}
    group_default_tau = DEFAULT_TAU
    params_path = group_dir / settings.PARAMS_FILENAME
    if not params_path.exists():
        return param_lookup, group_default_tau
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
            for p in params:
                if p.get("group_default"):
                    group_default_tau = p["tau"]
                elif "template" in p:
                    if "map_size" in p and "is_random" in p:
                        key = (p["template"], str(p["map_size"]), str(p["is_random"]))
                        param_lookup[key] = p["tau"]
                    else:
                        param_lookup[p["template"]] = p["tau"]
    except Exception:
        pass
    return param_lookup, group_default_tau

def compute_win_probability(mu_a: float, mu_b: float, tau: float = DEFAULT_TAU) -> float:
    """P(A beats B) = sigmoid((μ_A - μ_B) / tau)"""
    try:
        return 1.0 / (1.0 + math.exp(-(mu_a - mu_b) / tau))
    except OverflowError:
        return 0.0 if (mu_a - mu_b) < 0 else 1.0

def load_ratings(filepath: Path) -> dict:
    """Load player ratings (mu and sigma)."""
    ratings = {}
    if not filepath.exists():
        return ratings
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('mu') or not row.get('sigma'):
                continue
            # Handle different key names between WHR/TS
            lcb = float(row.get('lcb') or row.get('rating_lcb', 0))
            ratings[row['player_id']] = {
                'mu': float(row['mu']),
                'sigma': float(row['sigma']),
                'lcb': lcb
            }
    return ratings

def evaluate_model(group_name: str, filter_top_n: int = None):
    group_dir = settings.get_group_dir(group_name)
    ratings_file = group_dir / settings.RATINGS_FILENAME
    matches_file = group_dir / "matches.csv"
    
    ratings = load_ratings(ratings_file)
    if not ratings:
        print(f"    [!] No ratings found at {ratings_file}")
        return None

    print(f"\n[*] Evaluating Group: {group_name}" + (f" (Top {filter_top_n} Matchups Only)" if filter_top_n else "") + "...")
    
    # Identify elite pool if filtering is requested
    if filter_top_n:
        # Filter for eligible players first
        eligible_ids = [pid for pid, r in ratings.items() if r.get('lcb', 0) > 0]
        sorted_ids = sorted(eligible_ids, key=lambda x: ratings[x]['lcb'], reverse=True)
        elite_ids = set(sorted_ids[:filter_top_n])
        print(f"    Restricting to matches between top {len(elite_ids)} elite players.")
    else:
        elite_ids = None

    param_lookup, group_default_tau = load_param_lookup(group_dir)
    
    total_abs_error = 0.0
    total_log_loss = 0.0
    total_matches = 0
    correct_predictions = 0
    z_scores = []
    
    if not matches_file.exists():
        print(f"    [!] Group matches not found at {matches_file}")
        return None

    with open(matches_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_id, p2_id = row['p1_id'], row['p2_id']
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            template = row.get('template')
            
            if p1_s == 1 and p2_s == 0: winner = p1_id
            elif p1_s == 0 and p2_s == 1: winner = p2_id
            else: continue
            
            if p1_id not in ratings or p2_id not in ratings: continue
            
            # Elite-only filter
            if elite_ids and (p1_id not in elite_ids or p2_id not in elite_ids):
                continue
            
            # Get template specific TAU with fallbacks
            size = str(row.get('map_size', ''))
            rand = str(row.get('is_random', '0'))
            lookup_triple = (template, size, rand)
            
            tau = group_default_tau
            if lookup_triple in param_lookup:
                tau = param_lookup[lookup_triple]
            elif template in param_lookup:
                tau = param_lookup[template]
            
            mu1, mu2 = ratings[p1_id]['mu'], ratings[p2_id]['mu']
            prob = compute_win_probability(mu1, mu2, tau) # Prob that p1 wins
            outcome = 1.0 if winner == p1_id else 0.0
            
            # Accuracy metric: Did we correctly predict the winner (>50% prob)?
            if (prob > 0.5 and outcome == 1.0) or (prob < 0.5 and outcome == 0.0):
                correct_predictions += 1
            
            abs_error = abs(outcome - prob)
            eps = 1e-15
            loss = -(outcome * math.log(max(prob, eps)) + (1.0 - outcome) * math.log(max(1.0 - prob, eps)))
            
            total_abs_error += abs_error
            total_log_loss += loss
            total_matches += 1
            
            sd = math.sqrt(prob * (1.0 - prob))
            if sd > 0:
                z_scores.append(abs(outcome - prob) / sd)

    metrics = {
        'name': group_name,
        'matches': total_matches,
        'mae': total_abs_error / total_matches if total_matches > 0 else 0,
        'accuracy': correct_predictions / total_matches if total_matches > 0 else 0,
        'log_loss': total_log_loss / total_matches if total_matches > 0 else 0,
        'avg_z': np.mean(z_scores) if z_scores else 0,
        'std_z': np.std(z_scores) if z_scores else 0
    }
    return metrics

def main():
    filter_n = 1000 if "--elite-only" in sys.argv else None
    
    # Load groups from template_groups.json
    if not settings.TEMPLATE_GROUPS_JSON.exists():
        print(f"Error: {settings.TEMPLATE_GROUPS_JSON} not found.")
        return

    with open(settings.TEMPLATE_GROUPS_JSON, 'r') as f:
        groups = json.load(f)
    
    results = []
    for group_name in groups.keys():
        m = evaluate_model(group_name, filter_top_n=filter_n)
        if m: results.append(m)
    
    if not results: return
    
    print("\n" + "="*85)
    print(f"{'Group':25} | {'MAE':8} | {'Acc%':8} | {'LogLoss':8} | {'Mean Z':8} | {'Std Z':8}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:25} | {r['mae']:.4f} | {r['accuracy']*100:6.1f}% | {r['log_loss']:.4f} | {r['avg_z']:.4f} | {r['std_z']:.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
