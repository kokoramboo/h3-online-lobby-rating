import json
import csv
import subprocess
import math
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import settings

# Configuration
MATCHES_FILE = settings.MATCHES_CSV
GROUPS_FILE = settings.TEMPLATE_GROUPS_JSON
DATA_DIR = settings.WORKING_DIR
RATINGS_SCRIPT = settings.RATINGS_SCRIPT

def backup_outputs(group_dir):
    """Copy existing output files to .backup extension."""
    files_to_backup = [
        settings.RATINGS_FILENAME,
        settings.PRIORS_FILENAME,
        settings.PARAMS_FILENAME,
        "ratings_50plus.csv"
    ]
    for filename in files_to_backup:
        path = group_dir / filename
        if path.exists():
            backup_path = path.with_suffix(path.suffix + ".backup")
            import shutil
            shutil.copy2(path, backup_path)
            print(f"[*] Backed up {filename} to {backup_path.name}")

def load_groups():
    with open(GROUPS_FILE, 'r') as f:
        return json.load(f)

def filter_matches(group_name, templates):
    group_dir = settings.get_group_dir(group_name)
    output_path = group_dir / "matches.csv"
    temp_path = output_path.with_suffix('.tmp')
    count = 0
    try:
        with open(MATCHES_FILE, 'r') as f_in, open(temp_path, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row.get('template') in templates:
                    writer.writerow(row)
                    count += 1
        # Atomic swap
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e
    return output_path, count

def run_rating_step(matches_path, output_path, priors_path=None, tau=None, no_lcb=False, params_path=None, priors_out=None):
    cmd = ["python3", "-u", RATINGS_SCRIPT, "--matches", str(matches_path), "--output", str(output_path)]
    if priors_path:
        cmd.extend(["--priors", str(priors_path)])
    if priors_out:
        cmd.extend(["--priors-output", str(priors_out)])
    if tau:
        cmd.extend(["--tau", str(tau)])
    if no_lcb:
        cmd.append("--no-lcb")
    if params_path:
        cmd.extend(["--params", str(params_path)])
    
    print(f"[*] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def calibrate_tau_granular(matches_path, ratings_path, templates):
    print(f"[*] Calibrating granular TAU for group templates...")
    
    # Load ratings
    ratings = {}
    with open(ratings_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('mu'): continue
            ratings[row['player_id']] = {'mu': float(row['mu'])}
    
    # Load all group matches into memory
    all_matches = []
    with open(matches_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1_id, p2_id = row['p1_id'], row['p2_id']
            p1_s, p2_s = int(row.get('p1_status', 0) or 0), int(row.get('p2_status', 0) or 0)
            if p1_s == 1 and p2_s == 0: winner = p1_id
            elif p1_s == 0 and p2_s == 1: winner = p2_id
            else: continue
            
            if p1_id not in ratings or p2_id not in ratings: continue
            all_matches.append({
                'p1_id': p1_id, 'p2_id': p2_id, 'winner': winner, 
                'template': row.get('template'),
                'map_size': row.get('map_size', ''),
                'is_random': row.get('is_random', '0')
            })

    def compute_win_prob(mu_a, mu_b, tau):
        try:
            return 1.0 / (1.0 + math.exp(-(mu_a - mu_b) / tau))
        except OverflowError:
            return 0.0 if (mu_a - mu_b) < 0 else 1.0

    def run_sweep(matches):
        pair_stats = defaultdict(lambda: {'total': 0, 'p1_wins': 0})
        for m in matches:
            p_small, p_large = (m['p1_id'], m['p2_id']) if m['p1_id'] < m['p2_id'] else (m['p2_id'], m['p1_id'])
            pair = (p_small, p_large)
            if m['winner'] == p_small: pair_stats[pair]['p1_wins'] += 1
            pair_stats[pair]['total'] += 1

        if not pair_stats: return 5.50

        # Vectorized Sweep
        pairs = list(pair_stats.keys())
        diff_mus = np.array([ratings[p[0]]['mu'] - ratings[p[1]]['mu'] for p in pairs])
        p1_wins = np.array([pair_stats[p]['p1_wins'] for p in pairs])
        totals = np.array([pair_stats[p]['total'] for p in pairs])
        
        taus = np.arange(2.5, 9.5, 0.25)
        best_tau = 5.50
        min_log_loss = float('inf')
        eps = 1e-15

        for tau in taus:
            # Sigmoid: 1 / (1 + exp(-delta/tau))
            probs = 1.0 / (1.0 + np.exp(-diff_mus / tau))
            # Log-Loss: -(y*log(p) + (n-y)*log(1-p))
            losses = -(p1_wins * np.log(np.maximum(probs, eps)) + 
                       (totals - p1_wins) * np.log(np.maximum(1.0 - probs, eps)))
            avg_loss = np.sum(losses) / np.sum(totals)
            
            if avg_loss < min_log_loss:
                min_log_loss = avg_loss
                best_tau = tau
                
        return float(best_tau)

    results = []
    
    # 1. Add group-wide optimal TAU for fallback
    print(f"    [*] Calibrating group-wide fallback TAU...")
    group_tau = run_sweep(all_matches)
    print(f"        Group Fallback TAU: {group_tau:.2f}")
    results.append({"group_default": True, "tau": group_tau})

    # 2. Group matches by (template, map_size, is_random)
    units = defaultdict(list)
    for m in all_matches:
        key = (m['template'], m['map_size'], m['is_random'])
        units[key].append(m)

    for (t, s, r), u_matches in units.items():
        if len(u_matches) >= 1000:
            triple_key = f"{t}_{s}_{r}"
            print(f"    [*] Calibrating unit: {triple_key} ({len(u_matches)} matches)")
            tau = run_sweep(u_matches)
            print(f"        Optimal TAU: {tau:.2f}")
            results.append({
                "template": t,
                "map_size": s,
                "is_random": r,
                "tau": tau
            })

    # 3. Add per-template calibrations (if >= 1000 total)
    t_groups = defaultdict(list)
    for m in all_matches: t_groups[m['template']].append(m)
    
    for t, t_matches in t_groups.items():
        if len(t_matches) >= 1000:
            print(f"    [*] Calibrating template default: {t} ({len(t_matches)} matches)")
            tau = run_sweep(t_matches)
            print(f"        Optimal TAU: {tau:.2f}")
            results.append({
                "template": t,
                "tau": tau
            })
        else:
            print(f"    [*] Using group fallback for template: {t}")
            # No need to add entry, will fallback to group_default

    return results

def main():
    parser = argparse.ArgumentParser(description="H3 Online Lobby Rating System - Per-Group Pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-run of all steps even if output files exist")
    parser.add_argument("--priors", type=str, help="Global priors file to bootstrap all groups")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    
    for group_name, templates in groups.items():
        print(f"\n" + "="*60)
        print(f"PROCESSING GROUP: {group_name}")
        print("="*60)
        
        group_dir = settings.get_group_dir(group_name)
        backup_outputs(group_dir)
        
        # 1. Filter
        matches_path, count = filter_matches(group_name, templates)
        print(f"[*] Extracted {count} matches.")
        if count < 1000:
            print(f"Skip {group_name}: Too few matches ({count} < 1000).")
            continue
            
        group_priors = group_dir / settings.PRIORS_FILENAME
        
        # Determine refinement source
        if args.priors and Path(args.priors).exists():
            print(f"[*] Using GLOBAL priors provided: {args.priors}")
            refinement_out = Path(args.priors)
        elif group_priors.exists() and not args.force:
            print(f"[*] Found group-specific priors: {group_priors}. Skipping refinement.")
            refinement_out = group_priors
        else:
            if args.priors and not Path(args.priors).exists():
                print(f"[!] Warning: Global priors file {args.priors} not found. Falling back to full refinement.")
            if args.force and group_priors.exists():
                print(f"[*] --force enabled: Ignoring local group priors, performing full refinement.")
            # 2. Phase 1 (Initial pass to establish preliminary ratings)
            p1_out = group_dir / "phase1.csv"
            if not p1_out.exists() or args.force:
                run_rating_step(matches_path, p1_out, priors_path=None, no_lcb=True)
            else:
                print(f"[*] Skipping Phase 1: {p1_out} already exists.")
            
            # 3. Phase 2 (Refinement pass to improve convergence)
            p2_out = group_dir / "phase2.csv"
            if not p2_out.exists() or args.force:
                run_rating_step(matches_path, p2_out, priors_path=p1_out, no_lcb=True)
            else:
                print(f"[*] Skipping Phase 2: {p2_out} already exists.")
            refinement_out = p2_out
        
        # 4. Calibrate granularly
        template_params = calibrate_tau_granular(matches_path, refinement_out, templates)
        params_json_path = group_dir / settings.PARAMS_FILENAME
        temp_params_path = params_json_path.with_suffix('.tmp')
        try:
            with open(temp_params_path, 'w') as f:
                json.dump(template_params, f, indent=4)
            temp_params_path.replace(params_json_path)
        except Exception as e:
            if temp_params_path.exists():
                temp_params_path.unlink()
            raise e
        
        # 5. Phase 3 (Final with granular calibrated Taus)
        final_out = group_dir / settings.RATINGS_FILENAME
        priors_out = group_dir / settings.PRIORS_FILENAME
        run_rating_step(matches_path, final_out, priors_path=refinement_out, params_path=params_json_path, priors_out=priors_out)
        
    print(f"\n[!] All groups processed.")

if __name__ == "__main__":
    main()
