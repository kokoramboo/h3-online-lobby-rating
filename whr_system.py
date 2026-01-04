"""
Whole-History Rating (WHR) System for H3 Match Data
Based on Rémi Coulom's WHR algorithm.

Key Features:
- Global optimization (Maximum A Posteriori)
- Wiener process for skill drift
- Backward propagation (Future games inform past ratings)
"""

import csv
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import optimize

# Constants
W2 = 0.005  # Variance drift per day (same as our TrueSkill TAU)
INITIAL_MU = 0.0  # Log-space rating
DEFAULT_SIGMA = 1.0

class WHRPlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        # days_from_start -> [match_index, ...]
        self.days_to_matches = defaultdict(list)
        self.days = [] # Unique days player played
        self.day_to_idx = {} # day -> index in self.days
        self.ratings = None # Array of ratings per active day
        self.uncertainties = None # Array of sigma values per active day
        
    def add_match(self, day, match_idx):
        self.days_to_matches[day].append(match_idx)
        
    def setup(self):
        self.days = sorted(self.days_to_matches.keys())
        self.day_to_idx = {day: i for i, day in enumerate(self.days)}
        self.ratings = np.zeros(len(self.days)) # Start at 0
        self.uncertainties = np.ones(len(self.days)) # Default sigma

class WHRSystem:
    def __init__(self, w2=W2):
        self.w2 = w2
        self.players = {} # id -> WHRPlayer
        self.matches = [] # List of (winner_id, loser_id, day)
        
    def add_match(self, p1_id, p2_id, winner_id, start_time_str):
        # Parse time to days from an epoch
        try:
            dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        except:
            dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
            
        day = (dt - datetime(2020, 1, 1, tzinfo=timezone.utc)).days
        
        if p1_id not in self.players: self.players[p1_id] = WHRPlayer(p1_id)
        if p2_id not in self.players: self.players[p2_id] = WHRPlayer(p2_id)
        
        match_idx = len(self.matches)
        is_draw = (winner_id is None)
        self.matches.append({
            'winner': winner_id, 
            'loser': (p1_id if winner_id != p1_id else p2_id) if winner_id else None, 
            'p1': p1_id,
            'p2': p2_id,
            'day': day,
            'is_draw': is_draw
        })
        
        self.players[p1_id].add_match(day, match_idx)
        self.players[p2_id].add_match(day, match_idx)

    def load_matches(self, filepath: Path):
        print(f"[*] Loading matches for WHR from {filepath}...")
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic duration filter as in rating_system.py
                if int(row.get('duration', 0) or 0) < 600:
                    continue
                
                p1_status = int(row.get('p1_status', 0) or 0)
                p2_status = int(row.get('p2_status', 0) or 0)
                
                if p1_status == 1 and p2_status == 0:
                    self.add_match(row['p1_id'], row['p2_id'], row['p1_id'], row['start_time'])
                elif p1_status == 0 and p2_status == 1:
                    self.add_match(row['p1_id'], row['p2_id'], row['p2_id'], row['start_time'])
                elif p1_status == p2_status:
                    # Draw
                    self.add_match(row['p1_id'], row['p2_id'], None, row['start_time'])
        
        print(f"    Loaded {len(self.matches)} matches, {len(self.players)} players.")
        for p in self.players.values():
            p.setup()

    def iterate(self):
        """Perform one Newton-Raphson iteration over all players."""
        total_delta = 0
        for p_id, player in self.players.items():
            if not player.days: continue
            
            # 1. Gradient of Log-Prior (Wiener Process)
            # 2. Gradient of Log-Likelihood (Bradley-Terry)
            # 3. Hessian (Second Derivatives)
            
            # For simplicity in this implementation, we optimize one player at a time
            # assuming others are fixed (Coordinate Descent / Iterative Refinement)
            
            n = len(player.days)
            grad = np.zeros(n)
            hess = np.zeros((n, n))
            
            # Likelihood Contribution
            for i, day in enumerate(player.days):
                r_i = player.ratings[i]
                for m_idx in player.days_to_matches[day]:
                    match = self.matches[m_idx]
                    
                    # Opponent's rating at that time
                    opp_id = match['p1'] if match['p1'] != p_id else match['p2']
                    opp = self.players[opp_id]
                    
                    # Optimized lookup
                    opp_day_idx = opp.day_to_idx[day]
                    r_j = opp.ratings[opp_day_idx]
                    
                    # Bradley-Terry: P = e^ri / (e^ri + e^rj)
                    exp_i = math.exp(r_i)
                    exp_j = math.exp(r_j)
                    p_win = exp_i / (exp_i + exp_j)
                    
                    if match['is_draw']:
                        # Draw contributes 0.5 win and 0.5 loss
                        grad[i] += (0.5 - p_win)
                    elif match['winner'] == p_id:
                        grad[i] += (1.0 - p_win)
                    else:
                        grad[i] += (-p_win)
                        
                    hess[i, i] -= p_win * (1.0 - p_win)
            
            # Prior Contribution (Wiener Process: (ri+1 - ri)^2 / (2 * w2 * delta_t))
            for i in range(n - 1):
                dt = player.days[i+1] - player.days[i]
                sigma2 = self.w2 * dt
                diff = player.ratings[i+1] - player.ratings[i]
                
                # Grad: d/dri = diff/sigma2, d/dri+1 = -diff/sigma2
                grad[i] += diff / sigma2
                grad[i+1] -= diff / sigma2
                
                # Hessian
                hess[i, i] -= 1.0 / sigma2
                hess[i+1, i+1] -= 1.0 / sigma2
                hess[i, i+1] += 1.0 / sigma2
                hess[i+1, i] += 1.0 / sigma2
                
            # Solve Newton Step: delta = -H^-1 * G
            try:
                # Add tiny epsilon to diagonal for stability
                diag_indices = np.diag_indices(n)
                hess[diag_indices] -= 1e-9
                
                # Inverse Hessian diagonal provides the variance
                # Var(r) = -1 / diag(H) -- simplified estimation
                # Better: player.uncertainties = np.sqrt(np.diag(np.linalg.inv(-hess)))
                # But inv() is expensive. Let's use it sparingly or only at the end.
                
                delta = np.linalg.solve(hess, -grad)
                
                # Dampen updates to prevent oscillation
                player.ratings += delta * 0.7
                total_delta += np.sum(np.abs(delta))
                
                # Update uncertainties (only roughly for now, or at end)
                if n == 1:
                    player.uncertainties[0] = math.sqrt(-1.0 / hess[0, 0])
                else:
                    # Rough estimate
                    player.uncertainties = np.sqrt(-1.0 / np.diag(hess))
                    
            except np.linalg.LinAlgError:
                continue
                
        return total_delta

    def run(self, iterations=10):
        print(f"[*] Starting WHR iterations...")
        for i in range(iterations):
            start = time.time()
            delta = self.iterate()
            elapsed = time.time() - start
            print(f"    Iteration {i+1}/{iterations} | Delta: {delta:.2f} | Time: {elapsed:.1f}s")
            if delta < 1.0:
                print("    Converged.")
                break

    def save_results(self, output_path: Path, names: dict):
        print(f"[*] Saving WHR results to {output_path}...")
        results = []
        for p_id, player in self.players.items():
            if not player.days: continue
            
            # Final rating is the latest estimate
            mu = player.ratings[-1]
            sigma = player.uncertainties[-1]
            lcb = mu - 2.57 * sigma # 99% confidence
            g = sum(len(m) for m in player.days_to_matches.values())
            
            results.append({
                'player_id': p_id,
                'name': names.get(p_id, p_id),
                'mu': mu,
                'sigma': sigma,
                'lcb': lcb,
                'games': g
            })
            
        # --- Elite-Anchored Normalization ---
        # 1. Get Top 1000 by LCB
        results.sort(key=lambda x: x['lcb'], reverse=True)
        elite_pool = results[:1000]
        avg_mu_elite = np.mean([p['mu'] for p in elite_pool])
        print(f"Elite Pool (Top 1000) Average Mu: {avg_mu_elite:.4f}")
        
        # 2. Normalize every player relative to elite avg
        for p in results:
            lcb = p['lcb']
            try:
                # scale = 0.8 (Calibrated to keep Top 1000 spread professional)
                diff = lcb - avg_mu_elite
                if diff > 10: norm_rating = 100.0
                elif diff < -10: norm_rating = 0.0
                else:
                    norm_rating = 100.0 / (1.0 + math.exp(-diff / 0.8))
            except OverflowError:
                norm_rating = 100.0 if (lcb - avg_mu_elite) > 0 else 0.0
            p['norm_rating'] = round(norm_rating, 2)
            
            # Round for saving
            p['mu'] = round(p['mu'], 4)
            p['sigma'] = round(p['sigma'], 4)
            p['lcb'] = round(p['lcb'], 4)

        # Filter for >= 50 games for the final CSV saving (optional, but keep it for leaderboard)
        leaderboard = [p for p in results if p['games'] >= 50]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['player_id', 'name', 'mu', 'sigma', 'lcb', 'norm_rating', 'games'])
            writer.writeheader()
            writer.writerows(leaderboard)

def main():
    data_dir = Path(__file__).parent / 'data'
    whr = WHRSystem()
    whr.load_matches(data_dir / 'matches_jc_filtered.csv')
    
    # Load names
    names = {}
    with open(data_dir / 'players.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[row['player_id']] = row['nickname']
            
    whr.run(iterations=20)
    whr.save_results(data_dir / 'ratings_jc_whr.csv', names)

if __name__ == "__main__":
    main()
