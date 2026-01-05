# H3 Online Lobby Rating System

Bayesian rating system for Heroes of Might and Magic III (HotA) Online Lobby match data.

## Key Features
- **Bayesian Rating Inference**: Uses a Bayesian approach to estimate player skill (mu) and uncertainty (sigma).
- **Granular TAU Calibration**: Automatically calibrates the system's sensitivity (Dynamic BETA) based on template randomness(template, map/ size, random/not random game)
- **Elite-Anchored Win Rates**: Normalizes expected win rates against a pool of the top 1000 players.
- **Temporal Skill Drift**: Accounts for skill degradation over time (calibrated to ~0.005 sigma/day).

## How it Works

### 1. Leaderboard Eligibility
To ensure the leaderboard represents established players, two filters are applied:
- **Connectivity (5-Core)**: You must have played matches against at least 5 **different** players who are themselves well-connected. This prevents "rating farming" in isolated groups or against alt accounts.
- **Experience (50+ Games)**: While your rating is calculated from game 1, you only appear on the public leaderboard after completing 50 valid matches.

> [!NOTE]
> This project implements Bayesian skill inference based on the algorithm developed by Microsoft Research (similar to the logic used by TrueSkill™). TrueSkill™ is a trademark of Microsoft Corporation. The official evaluation uses a **Conservative Skill Estimate (LCB)** which represents the "minimum guaranteed" skill level the system is 99% confident in.

### 1. Rating Orchestrator (`rating_pipeline.py`)
The main entry point for updating ratings. It orchestrates:
- **Filtering**: Segregates matches into template families (e.g., JC Family).
- **Calibration**: Runs a Maximum Likelihood Estimation (MLE) sweep to find the optimal `TAU` for specific map settings.
- **Inference**: Executes the Bayesian rating engine with temporal drift.
- **Resumption**: Automatically detects previous `priors.csv` to skip bootstrapping phases.

### 3. Rating Engine (`rating_system.py`)
The core mathematical engine that implements the TrueSkill logic and analytical win rate approximations.

### 4. Analysis Tools
- `calibrate_drift.py`: Calibrates the temporal skill drift parameter.
- `analyze_graph.py`: Visualizes the player connectivity graph (k-core analysis).
- `whr_system.py`: A Whole-History Rating (WHR) implementation used for benchmarking.
- `compute_fair_prices.py`: WIP. Computes gold handicap "fair prices" based on Elo/TrueSkill differentials.

## Data Formats

### Input: `matches.csv`
Required fields for rating processing:
- `p1_id`, `p2_id`: Unique player identifiers.
- `p1_status`, `p2_status`: Outcome (1 = Win, 0 = Loss).
- `start_time`: ISO8601 timestamp (e.g., `2024-01-01T12:00:00Z`).
- `template`: Template name (used for grouping).
- `map_size`: Numeric size code (e.g., `144` for XL, `108` for L).
- `is_random`: `1` if the matchup was random-random (used for calibration), `0` otherwise.
- `duration`: Match length in seconds (filtered < 600s by default).

### Output Data
The pipeline organizes results into group-specific subdirectories:

```text
data/groups/
└── <group_name>/               # e.g., JC_Family
    ├── matches.csv             # Filtered matches for this group
    ├── params.json             # Calibrated TAU parameters per template
    ├── phase1.csv/phase2.csv   # Intermediate bootstrapping priors
    ├── priors.csv              # Skill priors for the next incremental update
    ├── ratings.csv             # Master list (Internal - all players)
    └── ratings_50plus.csv      # Public Leaderboard (Target for website)
```

#### Field Definitions (`ratings_50plus.csv`)
Intuitive guide for players:
- `mu`: **Skill Level**. Higher is better. A 30.0 is an expert; 50.0 is elite.
- `sigma`: **Uncertainty**. Tells us how sure we are about your rating. Lower means you've played more games and your rank is "stable".
- `lcb`: **Conservative Skill Estimate**. This is the level the system is 99% sure about. It prevents new players from gaining high ranks by luck, as it requires more games to prove they belong there.
- `norm_rating`: **Power Score (0-1000)**. Represents your expected wins if you played 1000 matches against the Top 1000 elite players. For example, a rating of 776 means you are expected to win at least 776 games out of 1000 against the elite pool.
- `games`: Total games played in this category.

## Getting Started

1.  **Configure Environment**:
    *   Copy the sample configuration: `cp .env.sample .env`
    *   Adjust the paths in `.env` to point to your local data files.
2.  **Run the Pipeline**:
    *   Execute the full automated sync and rating update:
        ```bash
        ./run_pipeline.sh
        ```

## License
MIT
