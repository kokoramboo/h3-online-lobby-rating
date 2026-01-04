"""
H3 Online Lobby Rating System - Crawler
Fetch all matches from homm3freaks.ru

Usage:
    python crawler.py              # Start or resume crawling
    python crawler.py --status     # Show progress
"""

import csv
import json
import time
import signal
import sys
import os
from pathlib import Path
from datetime import datetime
import requests
from typing import Optional, List

# Configuration
BASE_URL = "https://homm3freaks.ru/v1"
DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "state.json"
PLAYERS_FILE = DATA_DIR / "players.csv"
MATCHES_FILE = DATA_DIR / "matches.csv"

# Rate limiting: 60 requests/min = 1 request per second
REQUEST_DELAY = 1.1  # seconds between requests

# Page sizes (based on testing)
PLAYERS_PAGE_SIZE = 50  # API limit
MATCHES_PAGE_SIZE = 100  # API returns max 100 per request

# Minimum matches threshold - skip players with fewer matches
MIN_MATCHES_THRESHOLD = 30

# Known snapshot IDs (templates)
# 1 = Jebus Cross (JC)
# We'll discover others by trying sequential IDs
SNAPSHOT_IDS = list(range(1, 20))  # Try snapshots 1-19

# Session headers
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) H3LobbyRatingSystem/1.0"
}

# CSV column definitions
PLAYER_COLUMNS = [
    "player_id", "nickname", "snapshot_id", "snapshot_name", "rank", 
    "rating", "matches", "wins", "losses", "draws", "winrate",
    "avg_match_time", "total_time_played"
]

MATCH_COLUMNS = [
    "match_id", "template", "map_size", "start_time", "end_time", "duration",
    "result", "is_random",
    "p1_id", "p1_name", "p1_color", "p1_town", "p1_hero", "p1_handicap", "p1_points", "p1_status",
    "p2_id", "p2_name", "p2_color", "p2_town", "p2_hero", "p2_handicap", "p2_points", "p2_status"
]


class RateLimiter:
    """Simple rate limiter with configurable delay."""
    
    def __init__(self, delay: float):
        self.delay = delay
        self.last_request = 0
    
    def wait(self):
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()


class State:
    """Manages crawler state for resumability."""
    
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()
    
    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "completed_snapshots": [],
            "completed_players": [],
            "current_snapshot": None,
            "current_snapshot_offset": 0,
            "last_updated": None
        }
    
    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def is_snapshot_complete(self, snapshot_id: int) -> bool:
        return snapshot_id in self.data["completed_snapshots"]
    
    def mark_snapshot_complete(self, snapshot_id: int):
        if snapshot_id not in self.data["completed_snapshots"]:
            self.data["completed_snapshots"].append(snapshot_id)
        self.data["current_snapshot"] = None
        self.data["current_snapshot_offset"] = 0
        self.save()
    
    def is_player_complete(self, player_id: int) -> bool:
        return player_id in self.data["completed_players"]
    
    def mark_player_complete(self, player_id: int):
        if player_id not in self.data["completed_players"]:
            self.data["completed_players"].append(player_id)
        self.save()
    
    def get_snapshot_offset(self, snapshot_id: int) -> int:
        if self.data["current_snapshot"] == snapshot_id:
            return self.data["current_snapshot_offset"]
        return 0
    
    def set_snapshot_offset(self, snapshot_id: int, offset: int):
        self.data["current_snapshot"] = snapshot_id
        self.data["current_snapshot_offset"] = offset
        self.save()


class H3Crawler:
    """Main crawler for homm3freaks.ru."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(REQUEST_DELAY)
        self.state = State(STATE_FILE)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.running = True
        
        # Progress counters
        self.total_players = 0
        self.total_matches = 0
        
        # Setup data directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files with headers if they don't exist
        self._init_csv(PLAYERS_FILE, PLAYER_COLUMNS)
        self._init_csv(MATCHES_FILE, MATCH_COLUMNS)
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Track saved match IDs per player to avoid duplicates on restart
        self.saved_matches = self._load_saved_match_ids()
        
        # Track saved player IDs to avoid duplicates in players.csv
        self.saved_players = self._load_saved_player_ids()
    
    def _init_csv(self, path: Path, columns: List[str]):
        """Initialize CSV file with headers if it doesn't exist."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
    
    def _load_saved_player_ids(self) -> set:
        """Load player IDs already in players.csv to avoid duplicates."""
        saved = set()
        if not PLAYERS_FILE.exists():
            return saved
        
        with open(PLAYERS_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    try:
                        saved.add(int(row[0]))
                    except ValueError:
                        pass
        return saved
    
    def _load_saved_match_ids(self) -> dict:
        """Load max match ID per player from matches.csv to avoid duplicates."""
        saved = {}  # {player_id: max_match_id}
        if not MATCHES_FILE.exists():
            return saved
        
        with open(MATCHES_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 9:
                    try:
                        match_id = int(row[0])
                        p1_id = int(row[8]) if row[8] else 0
                        p2_id = int(row[16]) if row[16] else 0
                        # Track max match ID for both players
                        saved[p1_id] = max(saved.get(p1_id, 0), match_id)
                        saved[p2_id] = max(saved.get(p2_id, 0), match_id)
                    except (ValueError, IndexError):
                        pass
        
        if saved:
            print(f"[*] Loaded {len(saved)} player match histories")
        return saved
    
    def _signal_handler(self, signum, frame):
        print("\n[!] Shutdown requested, saving state...")
        self.running = False
    
    def _request(self, method: str, url: str, **kwargs) -> Optional[dict]:
        """Make a rate-limited request with retry logic for unstable connections."""
        self.rate_limiter.wait()
        
        attempt = 0
        while self.running:  # Retry forever until connection restored or shutdown
            attempt += 1
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                
                if resp.status_code == 429:
                    wait_time = int(resp.headers.get("x-ratelimit-reset", 60))
                    print(f"[!] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if resp.status_code in (200, 201):
                    return resp.json()
                
                print(f"[!] HTTP {resp.status_code}: {url}")
                return None
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"[!] {type(e).__name__}, retry #{attempt} in 30s...")
                time.sleep(30)
            except Exception as e:
                print(f"[!] Error: {e}, retry #{attempt} in 30s...")
                time.sleep(30)
        
        return None  # Only reached if shutdown requested
    
    def get_players(self, snapshot_id: int, start: int, end: int) -> Optional[dict]:
        """Fetch players from a snapshot leaderboard."""
        url = f"{BASE_URL}/snapshot/{snapshot_id}/players?startRow={start}&endRow={end}"
        return self._request("POST", url, json={"season": 0})
    
    def get_matches(self, player_id: int) -> Optional[dict]:
        """Fetch all matches for a player with pagination."""
        all_matches = []
        stats = None
        offset = 0
        
        while True:
            url = f"{BASE_URL}/match?stats=1&startRow={offset}&endRow={offset + MATCHES_PAGE_SIZE}"
            data = self._request("POST", url, json={"playerId": str(player_id)})
            
            if data is None:
                break
            
            matches = data.get("rows", [])
            if stats is None:
                stats = data.get("stats", {})
            
            all_matches.extend(matches)
            
            total = stats.get("totalMatches", 0)
            if len(all_matches) >= total or len(matches) == 0:
                break
            
            offset += MATCHES_PAGE_SIZE
        
        return {"rows": all_matches, "stats": stats} if all_matches else None
    
    def save_players(self, snapshot_id: int, players: list):
        """Append players to CSV file (only new ones)."""
        with open(PLAYERS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for p in players:
                player_id = p.get("id")
                # Skip if player already in CSV
                if player_id in self.saved_players:
                    continue
                writer.writerow([
                    player_id,
                    p.get("nickname"),
                    snapshot_id,
                    p.get("snapshotName"),
                    p.get("rank"),
                    p.get("rating"),
                    p.get("matches"),
                    p.get("wins"),
                    p.get("losses"),
                    p.get("draws"),
                    p.get("winrate"),
                    p.get("averageMatchTime"),
                    p.get("timePlayed")
                ])
                self.saved_players.add(player_id)
    
    def save_matches(self, player_id: int, data: dict) -> int:
        """Append new matches to CSV file, skip already-saved ones."""
        matches = data.get("rows", [])
        max_saved = self.saved_matches.get(player_id, 0)
        new_count = 0
        
        with open(MATCHES_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for m in matches:
                match_id = m.get("id", 0)
                # Skip if already saved
                if match_id <= max_saved:
                    continue
                
                players = m.get("players", [])
                p1 = players[0] if len(players) > 0 else {}
                p2 = players[1] if len(players) > 1 else {}
                
                writer.writerow([
                    match_id,
                    m.get("templateName"),
                    m.get("mapSize"),
                    m.get("startAt"),
                    m.get("endAt"),
                    m.get("duration"),
                    m.get("matchResult"),
                    m.get("isRandomMatch"),
                    p1.get("playerId"),
                    p1.get("nickname"),
                    p1.get("color"),
                    p1.get("town"),
                    p1.get("hero"),
                    p1.get("handicap"),
                    p1.get("pointsGained"),
                    p1.get("playerStatus"),
                    p2.get("playerId"),
                    p2.get("nickname"),
                    p2.get("color"),
                    p2.get("town"),
                    p2.get("hero"),
                    p2.get("handicap"),
                    p2.get("pointsGained"),
                    p2.get("playerStatus")
                ])
                new_count += 1
        
        # Update tracking
        if matches:
            self.saved_matches[player_id] = max(m.get("id", 0) for m in matches)
        
        return new_count
    
    def _save_matches_with_id_set(self, data: dict, saved_ids: set) -> int:
        """Save matches using a set of IDs for proper duplicate detection."""
        matches = data.get("rows", [])
        new_count = 0
        
        with open(MATCHES_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for m in matches:
                match_id = m.get("id", 0)
                # Skip if already saved
                if match_id in saved_ids:
                    continue
                
                players = m.get("players", [])
                p1 = players[0] if len(players) > 0 else {}
                p2 = players[1] if len(players) > 1 else {}
                
                writer.writerow([
                    match_id,
                    m.get("templateName"),
                    m.get("mapSize"),
                    m.get("startAt"),
                    m.get("endAt"),
                    m.get("duration"),
                    m.get("matchResult"),
                    m.get("isRandomMatch"),
                    p1.get("playerId"),
                    p1.get("nickname"),
                    p1.get("color"),
                    p1.get("town"),
                    p1.get("hero"),
                    p1.get("handicap"),
                    p1.get("pointsGained"),
                    p1.get("playerStatus"),
                    p2.get("playerId"),
                    p2.get("nickname"),
                    p2.get("color"),
                    p2.get("town"),
                    p2.get("hero"),
                    p2.get("handicap"),
                    p2.get("pointsGained"),
                    p2.get("playerStatus")
                ])
                saved_ids.add(match_id)  # Update set to prevent duplicates
                new_count += 1
        
        return new_count
    
    def _fetch_missing_matches(self, player_id: int, saved_ids: set) -> int:
        """Fetch matches for a player, stopping early when we hit existing data."""
        total_added = 0
        offset = 0
        
        while True:
            url = f"{BASE_URL}/match?stats=1&startRow={offset}&endRow={offset + MATCHES_PAGE_SIZE}"
            data = self._request("POST", url, json={"playerId": str(player_id)})
            
            if data is None:
                break
            
            matches = data.get("rows", [])
            if len(matches) == 0:
                break
            
            # Count how many new matches in this page
            new_in_page = 0
            with open(MATCHES_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                for m in matches:
                    match_id = m.get("id", 0)
                    if match_id in saved_ids:
                        continue
                    
                    players = m.get("players", [])
                    p1 = players[0] if len(players) > 0 else {}
                    p2 = players[1] if len(players) > 1 else {}
                    
                    writer.writerow([
                        match_id,
                        m.get("templateName"),
                        m.get("mapSize"),
                        m.get("startAt"),
                        m.get("endAt"),
                        m.get("duration"),
                        m.get("matchResult"),
                        m.get("isRandomMatch"),
                        p1.get("playerId"),
                        p1.get("nickname"),
                        p1.get("color"),
                        p1.get("town"),
                        p1.get("hero"),
                        p1.get("handicap"),
                        p1.get("pointsGained"),
                        p1.get("playerStatus"),
                        p2.get("playerId"),
                        p2.get("nickname"),
                        p2.get("color"),
                        p2.get("town"),
                        p2.get("hero"),
                        p2.get("handicap"),
                        p2.get("pointsGained"),
                        p2.get("playerStatus")
                    ])
                    saved_ids.add(match_id)
                    new_in_page += 1
                    total_added += 1
            
            # Check if we've fetched all matches
            total = data.get("stats", {}).get("totalMatches", 0)
            if offset + len(matches) >= total:
                break
            
            offset += MATCHES_PAGE_SIZE
        
        return total_added
    
    def crawl_snapshot(self, snapshot_id: int):
        """Crawl all players from a snapshot."""
        if self.state.is_snapshot_complete(snapshot_id):
            print(f"[=] Snapshot {snapshot_id} already complete, skipping")
            return
        
        offset = self.state.get_snapshot_offset(snapshot_id)
        print(f"[>] Crawling snapshot {snapshot_id} from offset {offset}")
        
        while self.running:
            data = self.get_players(snapshot_id, offset, offset + PLAYERS_PAGE_SIZE)
            
            if data is None:
                print(f"[!] Snapshot {snapshot_id} returned no data")
                self.state.mark_snapshot_complete(snapshot_id)
                return
            
            players = data.get("rows", [])
            if not players:
                print(f"[+] Snapshot {snapshot_id} complete ({offset} players)")
                self.state.mark_snapshot_complete(snapshot_id)
                return
            
            self.save_players(snapshot_id, players)
            print(f"    Fetched players {offset}-{offset + len(players)}")
            
            # Process each player - already-complete players are skipped inside crawl_player
            all_done = True
            for player in players:
                if not self.running:
                    all_done = False
                    break
                
                # Skip players with fewer than MIN_MATCHES_THRESHOLD matches
                match_count = player.get("matches", 0)
                if match_count < MIN_MATCHES_THRESHOLD:
                    continue
                
                self.crawl_player(player["id"], player.get("nickname", "Unknown"))
            
            # Only advance offset if ALL players in this page were processed
            if all_done:
                offset += PLAYERS_PAGE_SIZE
                self.state.set_snapshot_offset(snapshot_id, offset)
    
    def crawl_player(self, player_id: int, nickname: str):
        """Crawl all matches for a player."""
        if self.state.is_player_complete(player_id):
            return
        
        data = self.get_matches(player_id)
        if data is None:
            print(f"    [!] Failed to get matches for {nickname} ({player_id})")
            return
        
        matches = data.get("rows", [])
        stats = data.get("stats", {})
        total = stats.get("totalMatches", len(matches))
        
        self.save_matches(player_id, data)
        self.state.mark_player_complete(player_id)
        
        # Update counters
        self.total_players += 1
        self.total_matches += len(matches)
        
        print(f"    [{nickname}] {len(matches)}/{total} matches | Total: {self.total_players} players, {self.total_matches} matches")
    
    def run(self):
        """Main crawl loop."""
        print("=" * 50)
        print("H3 Crawler - homm3freaks.ru")
        print("=" * 50)
        print(f"Rate limit: {REQUEST_DELAY}s between requests")
        print(f"Output: {PLAYERS_FILE.absolute()}")
        print(f"        {MATCHES_FILE.absolute()}")
        print("Press Ctrl+C to stop (state will be saved)")
        print()
        
        for snapshot_id in SNAPSHOT_IDS:
            if not self.running:
                break
            self.crawl_snapshot(snapshot_id)
        
        print()
        print("[*] Crawl finished!" if self.running else "[*] Crawl interrupted, state saved")
        self.show_status()
    
    def show_status(self):
        """Show current progress."""
        print()
        print("=== Status ===")
        print(f"Completed snapshots: {len(self.state.data['completed_snapshots'])}")
        print(f"Completed players: {len(self.state.data['completed_players'])}")
        
        # Count CSV rows
        if PLAYERS_FILE.exists():
            with open(PLAYERS_FILE) as f:
                player_rows = sum(1 for _ in f) - 1  # Exclude header
            print(f"Player rows in CSV: {player_rows}")
        
        if MATCHES_FILE.exists():
            with open(MATCHES_FILE) as f:
                match_rows = sum(1 for _ in f) - 1
            print(f"Match rows in CSV: {match_rows}")
        
        if self.state.data["last_updated"]:
            print(f"Last updated: {self.state.data['last_updated']}")
    
    def validate_and_update(self):
        """Validate local matches against API and download missing ones."""
        print()
        print("=== Match Validation Mode ===")
        print()
        
        # Load all players from players.csv
        players = {}  # {player_id: (nickname, expected_matches)}
        if not PLAYERS_FILE.exists():
            print("[!] No players.csv found")
            return
        
        with open(PLAYERS_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                expected = int(row.get('matches', 0) or 0)
                players[row['player_id']] = (row['nickname'], expected)
        
        print(f"[*] Loaded {len(players)} players from players.csv")
        
        # Count matches per player in matches.csv
        local_counts = {}  # {player_id: count}
        if MATCHES_FILE.exists():
            with open(MATCHES_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 17:
                        p1_id = row[8]
                        p2_id = row[16]
                        local_counts[p1_id] = local_counts.get(p1_id, 0) + 1
                        local_counts[p2_id] = local_counts.get(p2_id, 0) + 1
        
        print(f"[*] Counted matches for {len(local_counts)} players in matches.csv")
        
        # Pre-filter: only check players where local < expected
        candidates = []
        skipped = 0
        for player_id, (nickname, expected) in players.items():
            local = local_counts.get(player_id, 0)
            if local < expected:
                candidates.append((player_id, nickname, local, expected))
            else:
                skipped += 1
        
        print(f"[*] Pre-filter: {len(candidates)} candidates to check, {skipped} already complete")
        print()
        
        # Validate each player against API
        discrepancies = []
        updated = 0
        new_matches = 0
        validated_ok = 0
        
        # Load ALL saved match IDs for proper duplicate detection
        saved_match_ids = set()
        if MATCHES_FILE.exists():
            with open(MATCHES_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 1:
                        try:
                            saved_match_ids.add(int(row[0]))
                        except ValueError:
                            pass
        print(f"[*] Loaded {len(saved_match_ids)} unique match IDs from matches.csv")
        print()
        
        total_candidates = len(candidates)
        validated_ok = skipped  # Players we skipped are already OK
        
        for i, (player_id, nickname, local_count, expected) in enumerate(candidates):
            if not self.running:
                break
            
            # Progress every 50 candidates
            if (i + 1) % 50 == 0:
                print(f"[*] Progress: {i + 1}/{total_candidates} checked | "
                      f"Discrepancies: {len(discrepancies)} | New: {new_matches}")
            
            # Query API for actual total
            url = f"{BASE_URL}/match?stats=1&startRow=0&endRow=1"
            data = self._request("POST", url, json={"playerId": player_id})
            
            if data is None:
                continue
            
            api_count = data.get("stats", {}).get("totalMatches", 0)
            
            if api_count > local_count:
                missing = api_count - local_count
                discrepancies.append({
                    'player_id': player_id,
                    'nickname': nickname,
                    'local': local_count,
                    'api': api_count,
                    'missing': missing
                })
                
                # Download missing matches - optimized to stop when we hit existing data
                print(f"  [{nickname}] Local: {local_count}, API: {api_count} (+{missing} missing)")
                
                added = self._fetch_missing_matches(int(player_id), saved_match_ids)
                new_matches += added
                if added > 0:
                    updated += 1
                    print(f"    -> Downloaded {added} new matches")
                else:
                    print(f"    -> No new matches found (already have all)")
            else:
                validated_ok += 1
                print(f"  [{nickname}] OK - Local: {local_count}, API: {api_count}")
        
        # Summary
        print()
        print("=== Validation Complete ===")
        print(f"Total players: {len(players)}")
        print(f"Candidates checked: {total_candidates}")
        print(f"Pre-skipped (already complete): {skipped}")
        print(f"Discrepancies found: {len(discrepancies)}")
        print(f"Players updated: {updated}")
        print(f"New matches downloaded: {new_matches}")
        
        if discrepancies:
            print()
            print("Top 10 discrepancies:")
            for d in sorted(discrepancies, key=lambda x: -x['missing'])[:10]:
                print(f"  {d['nickname']}: local={d['local']}, api={d['api']}, missing={d['missing']}")


def main():
    if "--status" in sys.argv:
        crawler = H3Crawler()
        crawler.show_status()
    elif "--validate" in sys.argv:
        crawler = H3Crawler()
        crawler.validate_and_update()
    else:
        crawler = H3Crawler()
        crawler.run()


if __name__ == "__main__":
    main()

