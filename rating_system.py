"""
Bayesian Rating System for H3 Match Data

Implements:
- 5-core graph connectivity filtering
- Gaussian Bayesian skill inference
- Lower Confidence Bound (LCB) public ratings
- Per-template independence
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import trueskill
import argparse
import settings


# Configuration
MIN_DURATION_SECONDS = 600  # 10 minutes
K_CORE = 5
NUM_SAMPLES = 1000
LCB_QUANTILE = 0.01  # 1st percentile = 99% confidence
WIN_PROB_TAU = 5.50  # Calibrated from sweep
SKILL_DRIFT_PER_DAY = 0.005  # Calibrated from data sweep
# Bayesian engine settings (tau=0 because we use temporal drift instead)
trueskill.setup(mu=25.0, sigma=25.0/3, beta=25.0/6, tau=0.0, draw_probability=0.01)


def load_matches(filepath: Path, min_duration: int = MIN_DURATION_SECONDS, 
                 min_date: str = None, last_n_per_player: int = None) -> list:
    """Load matches, filtering by duration and optionally by date."""
    matches = []
    skipped_short = 0
    skipped_invalid = 0
    skipped_old = 0
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration = int(row.get('duration', 0) or 0)
            if duration < min_duration:
                skipped_short += 1
                continue
            
            # Filter by date if specified
            if min_date:
                start_time = row.get('start_time', '')[:10]  # Get YYYY-MM-DD
                if start_time < min_date:
                    skipped_old += 1
                    continue
            
            # Determine winner using p1_status/p2_status (1=win, 0=loss)
            p1_status = int(row.get('p1_status', 0) or 0)
            p2_status = int(row.get('p2_status', 0) or 0)
            
            is_draw = False
            if p1_status == 1 and p2_status == 0:
                winner, loser = row['p1_id'], row['p2_id']
            elif p1_status == 0 and p2_status == 1:
                winner, loser = row['p2_id'], row['p1_id']
            elif p1_status == p2_status:
                # Both 0 or both 1 is a draw
                winner, loser = row['p1_id'], row['p2_id']
                is_draw = True
            else:
                # Other inconsistent states
                skipped_invalid += 1
                continue
            
            matches.append({
                'match_id': row.get('match_id', ''),
                'p1_id': row['p1_id'],
                'p2_id': row['p2_id'],
                'winner': winner,
                'loser': loser,
                'is_draw': is_draw,
                'template': row.get('template', 'unknown'),
                'map_size': row.get('map_size', ''),
                'is_random': row.get('is_random', '0'),
                'duration': duration,
                'start_time': row.get('start_time', '')
            })
    
    skip_msg = f"skipped {skipped_short} short, {skipped_invalid} invalid"
    if min_date:
        skip_msg += f", {skipped_old} before {min_date}"
    print(f"Loaded {len(matches)} matches ({skip_msg})")
    
    # Filter to last N matches per player if specified
    if last_n_per_player:
        matches = filter_last_n_matches(matches, last_n_per_player)
    
    return matches


def filter_last_n_matches(matches: list, n: int) -> list:
    """Keep only matches where both players have this as one of their last N matches."""
    from collections import defaultdict
    
    # Group matches by player and sort by time
    player_matches = defaultdict(list)
    for i, m in enumerate(matches):
        player_matches[m['p1_id']].append((m['start_time'], i))
        player_matches[m['p2_id']].append((m['start_time'], i))
    
    # For each player, find their last N match indices
    valid_indices = set()
    for player_id, pm in player_matches.items():
        pm.sort(key=lambda x: x[0])  # Sort by time
        last_n = pm[-n:]  # Keep last N
        for _, idx in last_n:
            valid_indices.add(idx)
    
    # A match is kept if it's in the last N for BOTH players
    # Actually, we keep if it's in last N for at least one player (more inclusive)
    filtered = [m for i, m in enumerate(matches) if i in valid_indices]
    
    print(f"Filtered to last {n} matches per player: {len(filtered)} matches remaining")
    return filtered


def build_player_graph(matches: list) -> nx.Graph:
    """Build undirected graph where edge exists if players have >= 1 valid game."""
    G = nx.Graph()
    
    for match in matches:
        p1, p2 = match['p1_id'], match['p2_id']
        G.add_edge(p1, p2)
    
    print(f"Player graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_k_core(G: nx.Graph, k: int = K_CORE) -> set:
    """Compute k-core and return set of eligible player IDs."""
    core = nx.k_core(G, k=k)
    eligible = set(core.nodes())
    
    excluded = set(G.nodes()) - eligible
    print(f"{k}-core: {len(eligible)} eligible players, {len(excluded)} excluded")
    
    return eligible

def load_mu_priors(filepath: Path) -> dict:
    """Load mu values from existing ratings CSV to use as priors."""
    priors = {}
    if not filepath.exists():
        return priors
    
    print(f"[*] Loading mu priors from {filepath}...")
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('mu'):
                priors[row['player_id']] = float(row['mu'])
    print(f"    Loaded {len(priors)} priors.")
    return priors


def run_trueskill_inference(matches: list, eligible_players: set, priors: dict = None, params_path: Path = None) -> dict:
    """
    Run TrueSkill inference on matches with temporal skill drift.
    Returns dict: player_id -> trueskill.Rating
    """
    from datetime import datetime, timezone
    
    # Pre-parse times if they are strings
    # Load template parameters for dynamic BETA calculation
    template_params_list = []
    if params_path and params_path.exists():
        try:
            with open(params_path, "r") as f:
                template_params_list = json.load(f)
        except Exception:
            pass

    # Build lookup map for efficient access
    param_lookup = {}
    group_default_tau = WIN_PROB_TAU
    for p in template_params_list:
        if p.get("group_default"):
            group_default_tau = p["tau"]
        elif "template" in p:
            if "map_size" in p and "is_random" in p:
                key = (p["template"], str(p["map_size"]), str(p["is_random"]))
                param_lookup[key] = p["tau"]
            else:
                param_lookup[p["template"]] = p["tau"]
    
    if template_params_list:
        spec_count = len([k for k in param_lookup if isinstance(k, tuple)])
        temp_count = len([k for k in param_lookup if isinstance(k, str)])
        print(f"[*] Loaded template parameters: {spec_count} specific variants, {temp_count} template fallbacks (Group Default TAU: {group_default_tau:.2f})")

    # Pre-create TrueSkill engines for different templates to avoid global state changes
    # Scale: Default TAU=5.50 maps to Default BETA=4.166 (25/6)
    # Ratio: 4.166 / 5.50 = 0.7575
    BETA_RATIO = (25.0/6.0) / 5.50
    
    def create_engine(tau):
        return trueskill.TrueSkill(
            mu=25.0, 
            sigma=25.0/3, 
            beta=tau * BETA_RATIO, 
            tau=0.0, 
            draw_probability=0.01
        )

    engines = {} # template_name -> TrueSkill engine
    default_engine = create_engine(WIN_PROB_TAU)

    parsed_matches = []
    min_time = datetime.min.replace(tzinfo=timezone.utc)
    for m in matches:
        t_str = m.get('start_time', '')
        try:
            t = datetime.fromisoformat(t_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            t = min_time
        
        m_copy = m.copy()
        m_copy['dt'] = t
        parsed_matches.append(m_copy)

    sorted_matches = sorted(parsed_matches, key=lambda m: m['dt'])
    print(f"[*] Processing {len(sorted_matches)} matches in CHRONOLOGICAL order...")
    
    ratings = {}
    for pid in eligible_players:
        if priors and pid in priors:
            ratings[pid] = trueskill.Rating(mu=priors[pid])
        else:
            ratings[pid] = trueskill.Rating()
            
    last_time = {} 
    
    processed = 0
    import time
    start_time = time.time()

    for match in sorted_matches:
        if (processed + 1) % 100000 == 0:
            elapsed = time.time() - start_time
            rate = (processed + 1) / elapsed
            eta = (len(sorted_matches) - processed - 1) / rate
            print(f"  Progress: {processed + 1}/{len(sorted_matches)} ({rate:.0f}/s, ETA: {eta:.0f}s)")
            
        winner, loser = match['winner'], match['loser']
        
        if winner not in eligible_players or loser not in eligible_players:
            continue
            
        current_time = match['dt']
        
        for pid in [winner, loser]:
            if pid in last_time and current_time > min_time:
                delta = current_time - last_time[pid]
                delta_days = delta.total_seconds() / 86400.0
                if delta_days > 0:
                    new_sigma = math.sqrt(ratings[pid].sigma**2 + SKILL_DRIFT_PER_DAY * delta_days)
                    ratings[pid] = trueskill.Rating(mu=ratings[pid].mu, sigma=new_sigma)
        
        # Select engine based on template, size, and randomness
        # Fallback hierarchy: (template, size, random) -> (template) -> default
        template = match.get('template')
        size = str(match.get('map_size', ''))
        rand = str(match.get('is_random', '0'))
        
        lookup_triple = (template, size, rand)
        
        if lookup_triple not in engines:
            if lookup_triple in param_lookup:
                tau = param_lookup[lookup_triple]
            elif template in param_lookup:
                tau = param_lookup[template]
            else:
                tau = group_default_tau
            engines[lookup_triple] = create_engine(tau)
        
        engine = engines[lookup_triple]
        
        # Update ratings using the selected engine
        winner_rating, loser_rating = engine.rate_1vs1(
            ratings[winner], ratings[loser], drawn=match.get('is_draw', False)
        )
        ratings[winner] = winner_rating
        ratings[loser] = loser_rating
        
        last_time[winner] = current_time
        last_time[loser] = current_time
        processed += 1
    
    print(f"Processed {processed} matches with dynamic BETA based on template randomness")

    # Final Drift application: from last match to NOW
    now = datetime.now(timezone.utc)
    drift_applied = 0
    for pid in eligible_players:
        if pid in last_time:
            delta = now - last_time[pid]
            delta_days = delta.total_seconds() / 86400.0
            if delta_days > 0:
                new_sigma = math.sqrt(ratings[pid].sigma**2 + SKILL_DRIFT_PER_DAY * delta_days)
                ratings[pid] = trueskill.Rating(mu=ratings[pid].mu, sigma=new_sigma)
                drift_applied += 1
    
    print(f"Applied final temporal drift ({SKILL_DRIFT_PER_DAY}/day) to {drift_applied} players up to {now.isoformat()}")
    return ratings


def compute_win_probability(mu_a: float, mu_b: float) -> float:
    """Compute win probability of A beating B using calibrated sigmoid."""
    diff = mu_a - mu_b
    try:
        return 1.0 / (1.0 + math.exp(-diff / WIN_PROB_TAU))
    except OverflowError:
        return 0.0 if diff < 0 else 1.0




def compute_analytical_winrate(mu: float, sigma: float, mu_baseline: float, tau: float = WIN_PROB_TAU, lcb: bool = False) -> float:
    """
    Compute expected win rate analytically for Skill ~ N(mu, sigma^2) vs baseline mu.
    Uses Probit approximation for the expectation of a logistic sigmoid.
    """
    if lcb:
        # Lower Confidence Bound: compute win rate of the mu-space LCB
        mu_val = mu - 2.575 * sigma
        sigma_val = 0.0
    else:
        mu_val = mu
        sigma_val = sigma
        
    diff = mu_val - mu_baseline
    # Expectation of sigmoid(x/tau) where x ~ N(diff, sigma^2)
    # E[sigmoid] approx sigmoid( mu / sqrt(1 + (pi/8) * (sigma/tau)^2) )
    kappa = 1.0 / math.sqrt(1.0 + (math.pi / 8.0) * (sigma_val / tau)**2)
    return 1.0 / (1.0 + math.exp(-kappa * diff / tau))




def compute_all_ratings(matches: list, priors: dict = None, skip_lcb: bool = False, params_path: Path = None) -> list:
    """Main function to compute all ratings."""
    
    # Build graph and compute 5-core
    G = build_player_graph(matches)
    all_players = set(G.nodes())
    eligible_players = compute_k_core(G, K_CORE)
    
    # Run TrueSkill inference
    ratings = run_trueskill_inference(matches, eligible_players, priors=priors, params_path=params_path)
    
    # Skip LCB if requested (for speed in intermediate steps)
    if skip_lcb:
        print("[*] Skipping LCB computation as requested.")
        results = []
        for pid in eligible_players:
            results.append({
                'player_id': pid,
                'mu': round(ratings[pid].mu, 4),
                'sigma': round(ratings[pid].sigma, 4),
                'expected_win_rate': 0.5, # Dummy
                'rating_lcb': 0.0, # Dummy
                'rating_normalized': 0.0, # Dummy
                'status': 'eligible'
            })
        for pid in all_players - eligible_players:
             results.append({
                'player_id': pid,
                'status': 'insufficient connectivity'
            })
        return results

    # Process eligible players
    print(f"Computing ratings for {len(eligible_players)} eligible players...")
    
    # Workflow: 
    # 1. Find Top 1000 pool based on raw skill LCB (mu - 2.57*sigma)
    # 2. Compute avg_mu_elite
    # 3. Compute WR vs that avg analytically (Instant!)
    
    raw_results = []
    for pid in eligible_players:
        r = ratings[pid]
        raw_results.append({
            'player_id': pid,
            'mu': r.mu,
            'sigma': r.sigma,
            'skill_lcb': r.mu - 2.575 * r.sigma
        })
        
    # Find Elite Mean
    raw_results.sort(key=lambda x: x['skill_lcb'], reverse=True)
    elite_pool = raw_results[:1000]
    avg_mu_elite = np.mean([p['mu'] for p in elite_pool]) if elite_pool else 25.0
    print(f"[*] Elite Pool (Top 1000) Average Mu: {avg_mu_elite:.4f}")

    results = []
    for r in raw_results:
        # Analytical Win Rate and LCB versus Elite Average
        mean_wr = compute_analytical_winrate(r['mu'], r['sigma'], avg_mu_elite)
        lcb_wr = compute_analytical_winrate(r['mu'], r['sigma'], avg_mu_elite, lcb=True)
        
        results.append({
            'player_id': r['player_id'],
            'mu': round(r['mu'], 4),
            'sigma': round(r['sigma'], 4),
            'expected_win_rate': round(mean_wr, 4),
            'rating_lcb': round(lcb_wr, 4),
            'rating_normalized': round(lcb_wr * 1000, 0),
            'status': 'eligible',
            'component_size': len(eligible_players)
        })
    
    # Add excluded players
    excluded_players = all_players - eligible_players
    for player_id in excluded_players:
        results.append({
            'player_id': player_id,
            'mu': None,
            'sigma': None,
            'expected_win_rate': None,
            'rating_lcb': None,
            'rating_normalized': None,
            'status': 'insufficient connectivity',
            'component_size': None
        })
    
    # Sort by rating_normalized descending
    results.sort(key=lambda x: (x['status'] != 'eligible', -(x['rating_normalized'] or 0)))
    
    return results


def run_sanity_checks(results: list, matches: list) -> dict:
    """Run validation checks."""
    checks = {}
    
    eligible = [r for r in results if r['status'] == 'eligible']
    
    # Check 1: Higher μ should correlate with higher win rates
    mus = [r['mu'] for r in eligible]
    win_rates = [r['expected_win_rate'] for r in eligible]
    correlation = np.corrcoef(mus, win_rates)[0, 1]
    checks['mu_winrate_correlation'] = round(correlation, 4)
    checks['mu_winrate_correlation_pass'] = correlation > 0.9
    
    # Check 2: More games -> lower σ
    game_counts = defaultdict(int)
    for m in matches:
        game_counts[m['p1_id']] += 1
        game_counts[m['p2_id']] += 1
    
    eligible_with_games = [(r, game_counts.get(r['player_id'], 0)) for r in eligible]
    games = [g for _, g in eligible_with_games]
    sigmas = [r['sigma'] for r, _ in eligible_with_games]
    games_sigma_corr = np.corrcoef(games, sigmas)[0, 1]
    checks['games_sigma_correlation'] = round(games_sigma_corr, 4)
    checks['games_sigma_correlation_pass'] = games_sigma_corr < -0.3
    
    # Check 3: LCB < mean win rate for all players
    lcb_less_than_mean = all(
        r['rating_lcb'] <= r['expected_win_rate'] 
        for r in eligible
    )
    checks['lcb_less_than_mean'] = lcb_less_than_mean
    
    # Check 4: Summary stats
    checks['num_eligible'] = len(eligible)
    checks['num_excluded'] = len(results) - len(eligible) 
    checks['mean_sigma'] = round(np.mean([r['sigma'] for r in eligible]), 4)
    checks['mean_games'] = round(np.mean([game_counts.get(r['player_id'], 0) for r in eligible]), 2)
    return checks


def save_priors(results: list, output_path: Path):
    """Save simplified priors (player_id, mu) to CSV."""
    eligible = [r for r in results if r['status'] == 'eligible']
    fieldnames = ['player_id', 'mu']
    
    temp_path = output_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(eligible)
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e
    print(f"Saved {len(eligible)} priors to {output_path}")

def save_ratings(results: list, output_path: Path):
    """Save raw results to CSV."""
    fieldnames = ['player_id', 'mu', 'sigma', 'expected_win_rate', 'rating_lcb', 
                  'rating_normalized', 'status', 'component_size']
    
    temp_path = output_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        
        # Atomic swap
        temp_path.replace(output_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e
    
    print(f"Saved {len(results)} player ratings to {output_path}")


def save_ratings_filtered(results: list, matches: list, output_path: Path, 
                       names: dict, lobby_ratings: dict, min_games: int = 30):
    """Save final ratings CSV with game counts, filtered by min games."""
    from collections import defaultdict
    
    # Get eligible players
    eligible = set(r['player_id'] for r in results if r['status'] == 'eligible')
    
    # Count games and wins per player
    games = defaultdict(int)
    wins = defaultdict(int)
    for m in matches:
        p1, p2 = m['p1_id'], m['p2_id']
        if p1 in eligible and p2 in eligible:
            games[p1] += 1
            games[p2] += 1
            if m['winner'] == p1:
                wins[p1] += 1
            else:
                wins[p2] += 1
    
    # Write filtered CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'player_id', 'player_name', 'lcb_rating', 'lobby_rating', 
                         'mu', 'sigma', 'games', 'actual_winrate', 'expected_winrate'])
        
        rank = 0
        for r in results:
            pid = r['player_id']
            if r['status'] != 'eligible':
                continue
            
            g = games.get(pid, 0)
            if g < min_games:
                continue
            
            rank += 1
            actual_wr = wins.get(pid, 0) / g if g > 0 else 0
            writer.writerow([
                rank, pid, names.get(pid, pid), 
                round(r['rating_normalized'], 1),
                lobby_ratings.get(pid, 0),
                round(r['mu'], 2), round(r['sigma'], 2),
                g,
                round(actual_wr, 3), round(r['expected_win_rate'], 3)
            ])
    
    print(f"Saved {rank} players (≥{min_games} games) to {output_path}")


def load_player_info(filepath: Path) -> tuple:
    """Load player names and lobby ratings."""
    names = {}
    lobby_ratings = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[row['player_id']] = row['nickname']
            lobby_ratings[row['player_id']] = int(row.get('rating', 0) or 0)
    return names, lobby_ratings


def main():
    global WIN_PROB_TAU
    parser = argparse.ArgumentParser(description="""
H3 Online Lobby Rating System - Core Engine
Implements Bayesian skill inference (Gaussian Skill Rating) with temporal drift.
""")
    parser.add_argument("--matches", type=str, default="data/matches_jc_filtered.csv", help="Path to matches CSV")
    parser.add_argument("--output", type=str, default="data/ratings_jc.csv", help="Path to save ratings results")
    parser.add_argument("--priors-output", type=str, help="Path to save simplified priors for next run")
    parser.add_argument("--priors", type=str, help="Path to load priors from")
    parser.add_argument("--tau", type=float, default=5.50, help="WIN_PROB_TAU for sigmoid")
    parser.add_argument("--elite-anchor", type=int, default=1000, help="Number of top players for elite anchor")
    parser.add_argument("--min-games", type=int, default=50, help="Minimum games for 50plus leaderboard")
    parser.add_argument("--no-lcb", action="store_true", help="Skip slow LCB and normalization (useful for intermediate runs)")
    parser.add_argument("--params", type=str, help="Path to template_params.json")
    
    args = parser.parse_args()

    print(f"[*] H3 Rating Engine starting...")
    print(f"[*] Matches: {args.matches}")
    print(f"[*] Output: {args.output}")

    matches_path = Path(args.matches)
    output_path = Path(args.output)
    
    # Override globals if needed
    WIN_PROB_TAU = args.tau

    if not matches_path.exists():
        print(f"Error: {matches_path} not found")
        return

    matches = load_matches(matches_path)

    
    priors = None
    if args.priors:
        priors_path = Path(args.priors)
        if priors_path.exists():
            print(f"[*] Loading mu priors from {priors_path}...")
            priors = load_mu_priors(priors_path)
            print(f"    Loaded {len(priors)} priors.")
        else:
            print(f"Warning: Priors file {priors_path} not found. Starting fresh.")

    # Load player info for reporting and naming
    player_info_path = settings.PLAYERS_CSV
    names = {}
    lobby_ratings = {}
    if player_info_path.exists():
        names, lobby_ratings = load_player_info(player_info_path)

    params_path = Path(args.params) if args.params else settings.TEMPLATE_PARAMS_JSON
    results = compute_all_ratings(matches, priors=priors, skip_lcb=args.no_lcb, params_path=params_path)
    
    # Run sanity checks
    checks = {}
    if not args.no_lcb:
        checks = run_sanity_checks(results, matches)
    
    if checks:
        print("\n=== Sanity Checks ===")
        for key, value in checks.items():
            print(f"  {key}: {value}")
    
    # Save raw results
    save_ratings(results, output_path)

    # Save simplified priors if requested
    if args.priors_output:
        save_priors(results, Path(args.priors_output))
    
    # Save 50+ list (derive path from output name) if LCB was computed
    if not args.no_lcb:
        suffix = output_path.suffix
        base = str(output_path.with_suffix(''))
        plus_path = Path(f"{base}_{args.min_games}plus{suffix}")
        save_ratings_filtered(results, matches, plus_path, names=names, lobby_ratings=lobby_ratings, min_games=args.min_games)
    
    # Show top 20 players
    print("\n=== Top 20 Players (≥50 games) ===")
    eligible_list = [r for r in results if r['status'] == 'eligible']
    
    # Re-calculate games for display
    game_counts = defaultdict(int)
    for m in matches:
        game_counts[m['p1_id']] += 1
        game_counts[m['p2_id']] += 1

    rank = 0
    for r in eligible_list:
        pid = r['player_id']
        g = game_counts.get(pid, 0)
        if g < 50:
            continue
        rank += 1
        if rank > 20:
            break
        print(f"{rank:3}. {names.get(pid, pid):20} | Rating: {r['rating_normalized']:5.1f} | "
              f"μ={r['mu']:.1f}, σ={r['sigma']:.2f} | Games: {g}")


if __name__ == '__main__':
    main()

