# H3 Online Lobby Rating System

A high-performance, Bayesian rating system for Heroes of Might and Magic III (HotA) match data, designed for the homm3freaks.ru lobby.

## Key Features
- **TrueSkill™ Inference**: Uses a Bayesian approach to estimate player skill (mu) and uncertainty (sigma).
- **Granular TAU Calibration**: Automatically calibrates the system's sensitivity (Dynamic BETA) based on template randomness(template, map/ size, random/not random game)
- **Fast Analytical LCB**: Replaces slow Monte Carlo simulations with a high-precision Probit approximation for Lower Confidence Bound (LCB) ratings.
- **Elite-Anchored Win Rates**: Normalizes expected win rates against a pool of the top 1000 players.
- **Temporal Skill Drift**: Accounts for skill degradation over time (calibrated to ~0.005 sigma/day).
- **Production Ready**: Atomic file writing, automated backups, and per-group resume logic.

## Architecture

### 1. Data Collection (`crawler.py`)
Syncs match history from the lobby API into a local CSV database.

### 2. Rating Orchestrator (`rating_pipeline.py`)
The main entry point for updating ratings. It orchestrates:
- **Filtering**: Segregates matches into template families (e.g., JC Family).
- **Calibration**: Runs a Maximum Likelihood Estimation (MLE) sweep to find the optimal `TAU` for specific map settings.
- **Inference**: Executes the TrueSkill engine with temporal drift.
- **Resumption**: Automatically detects previous `priors.csv` to skip bootstrapping phases.

### 3. Rating Engine (`rating_system.py`)
The core mathematical engine that implements the TrueSkill logic and analytical win rate approximations.

### 4. Analysis Tools
- `calibrate_drift.py`: Calibrates the temporal skill drift parameter.
- `analyze_graph.py`: Visualizes the player connectivity graph (k-core analysis).
- `whr_system.py`: A Whole-History Rating (WHR) implementation used for benchmarking.
- `compute_fair_prices.py`: WIP. Computes gold handicap "fair prices" based on Elo/TrueSkill differentials.

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
