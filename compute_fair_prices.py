import pandas as pd
import numpy as np
import os
import settings

# Town Mapping
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

# Configuration
MATCHES_FILE = settings.MATCHES_CSV
RATINGS_FILE = settings.WORKING_DIR / "ratings_jc_whr.csv"
TEMPLATE = "Jebus Cross"

# Adjusted Beta: How much 1 gold affects log-odds.
# 1000 gold = 0.15 logit jump (~3.75% winrate change at 50%) is a standard pro estimate for high-level JC.
BETA_BID_FIXED = 0.00015 

def load_and_filter_data():
    if not os.path.exists(MATCHES_FILE):
        print(f"[!] {MATCHES_FILE} not found")
        return None
    
    print("[*] Reading CSV...")
    df = pd.read_csv(MATCHES_FILE)
    df = df[df['template'] == TEMPLATE]
    
    # Exclude random matches as requested by user
    print("[*] Excluding random matches (is_random=True)...")
    df = df[df['is_random'] == False]
    
    # Parse dates
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    # Date filters
    ns_cutoff = pd.Timestamp('2025-07-01', tz='UTC')
    other_cutoff = pd.Timestamp('2024-01-01', tz='UTC')
    
    is_ns = (df['p1_town'].isin([4, 7])) | (df['p2_town'].isin([4, 7]))
    df = df[((is_ns) & (df['start_time'] >= ns_cutoff)) | ((~is_ns) & (df['start_time'] >= other_cutoff))].copy()
    
    # Handicap filter
    df = df[df['p1_handicap'].between(-10000, 10000) & df['p2_handicap'].between(-10000, 10000)].copy()
    
    return df

def process_matches(df):
    if not os.path.exists(RATINGS_FILE):
        print(f"[!] {RATINGS_FILE} not found")
        return None
        
    ratings_df = pd.read_csv(RATINGS_FILE)
    top_1000_ids = set(ratings_df.sort_values('mu', ascending=False).head(1000)['player_id'])
    
    # Filter for top 1000 vs top 1000
    df = df[df['p1_id'].isin(top_1000_ids) & df['p2_id'].isin(top_1000_ids)].copy()
    
    # Standardize to Red/Blue perspective
    df['red_town'] = np.where(df['p1_color'] == 1, df['p1_town'], df['p2_town'])
    df['blue_town'] = np.where(df['p1_color'] == 0, df['p1_town'], df['p2_town'])
    
    # red_handicap: net gold for red.
    df['red_handicap'] = np.where(df['p1_color'] == 1, df['p1_handicap'], df['p2_handicap'])
    
    # red_won
    df['red_won'] = np.where(df['p1_color'] == 1, (df['p1_status'] == 1).astype(int), (df['p2_status'] == 1).astype(int))
    
    # Exclude draws/invalid
    df = df[df['p1_status'].isin([0, 1]) & df['p2_status'].isin([0, 1]) & (df['p1_status'] != df['p2_status'])]
    
    return df

def main():
    df = load_and_filter_data()
    if df is None: return
    df = process_matches(df)
    if df is None: return
    
    print(f"[*] Analyzing {len(df)} non-random matches between top 1000 players...")
    
    results = []
    
    # 1. Compute Mirror Matches for Color Bias
    mirrors = df[df['red_town'] == df['blue_town']]
    if not mirrors.empty:
        # Bias: Red pays Blue to be fair. If Red needs gold, bias is negative.
        # Fairness means Red_Won = LogitInv(b_bid * (Paid) + b0) = 0.5 => b_bid * Paid + b0 = 0
        avg_paid_mirror = -mirrors['red_handicap'].mean()
        wr_mirror = mirrors['red_won'].mean()
        if wr_mirror > 0 and wr_mirror < 1:
            color_bias = avg_paid_mirror + np.log(wr_mirror / (1-wr_mirror)) / BETA_BID_FIXED
        else:
            color_bias = avg_paid_mirror
        print(f"[*] Calculated Color Bias (Red Disadvantage): {color_bias:.1f} gold")
    else:
        color_bias = -1900 # Fallback
        print(f"[*] No mirror matches found, using fallback bias: {color_bias:.1f}")

    # 2. Compute Town Pairs
    town_ids = sorted(TOWN_MAP.keys())
    for r_id in town_ids:
        for b_id in town_ids:
            pair_mask = (df['red_town'] == r_id) & (df['blue_town'] == b_id)
            sub = df[pair_mask]
            
            if len(sub) < 2: continue # Threshold for stability
            
            avg_paid = -sub['red_handicap'].mean()
            wr = sub['red_won'].mean()
            
            if wr > 0 and wr < 1:
                logit_wr = np.log(wr / (1-wr))
                correction = logit_wr / BETA_BID_FIXED
            else:
                correction = 0
                
            fair_price = avg_paid + correction
            fair_price = round(fair_price, -2)
            
            results.append({
                "red_id": r_id,
                "blue_id": b_id,
                "Red Town": TOWN_MAP[r_id],
                "Blue Town": TOWN_MAP[b_id],
                "Fair Price": fair_price,
                "Winrate": round(wr, 3),
                "Matches": len(sub)
            })

    res_df = pd.DataFrame(results)
    
    # 3. Pivot and Present
    pivot_df = res_df.pivot(index='Red Town', columns='Blue Town', values='Fair Price')
    town_order = ["Castle", "Rampart", "Tower", "Inferno", "Necropolis", "Dungeon", "Stronghold", "Fortress", "Conflux", "Cove", "Factory"]
    present_towns = [t for t in town_order if t in pivot_df.index]
    pivot_df = pivot_df.reindex(index=present_towns, columns=present_towns)

    print("\nFair Price Table (Positive: Red pays Blue, Negative: Blue pays Red):")
    print(pivot_df)
    
    # 4. Pure Town Power Analysis
    # Red Castle vs Blue T = Strength_T - Strength_0 + Bias => Strength_T = Price - Bias
    # We use Castle as baseline (0)
    strengths = {'Castle': 0}
    for t in present_towns:
        if t == 'Castle': continue
        # Try both Red Castle vs Blue T and Blue Castle vs Red T
        sub_ct = res_df[(res_df['Red Town'] == 'Castle') & (res_df['Blue Town'] == t)]
        if not sub_ct.empty:
            strengths[t] = sub_ct['Fair Price'].iloc[0] - color_bias
        else:
            sub_tc = res_df[(res_df['Red Town'] == t) & (res_df['Blue Town'] == 'Castle')]
            if not sub_tc.empty:
                strengths[t] = color_bias - sub_tc['Fair Price'].iloc[0]
            else:
                strengths[t] = np.nan

    print("\nPure Town Power (Relative to Castle):")
    sorted_s = sorted(strengths.items(), key=lambda x: (x[1] if not np.isnan(x[1]) else -99999), reverse=True)
    for t, s in sorted_s:
        if not np.isnan(s):
            print(f"{t}: {s:+.1f}")
        else:
            print(f"{t}: Unknown")

    # Create sanitized filename
    safe_template = TEMPLATE.replace(" ", "_").replace(".", "_")
    output_path = settings.FAIR_PRICES_DIR / f"{safe_template}.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\n[*] Full results saved to {output_path}")

if __name__ == "__main__":
    main()
