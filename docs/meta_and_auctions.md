# H3 Competitive Economy: Auctions and Handicaps

This document details the strategic balancing phase that occurs before a competitive match, specifically on templates like **Jebus Cross**.

## The Balancing Loop
In professional play, standard town and side choices are balanced through a dynamic auction system. This ensures that even if one town is inherently stronger, the player wielding it must pay a "fair price" to the opponent.

### 1. The Roll (Random Pairs)
- Players generate a random town matchup (e.g., Castle vs Necropolis).
- This creates the initial state where one side may have a perceived advantage.

### 2. Town Auction
- Players auction for the **choice** of town.
- The winner of the auction chooses which town they want to play.
- The amount bid is paid from the winner's starting gold to the loser of the auction.
- This gold transfer is reflected in the `p1_handicap` and `p2_handicap` fields of the match data.

### 3. Side/Color Auction
- A second auction occurs for the **choice** of color (Red vs Blue).
- Again, the winner pays the bid amount to the loser.
- Blue typically receives more time and can counter-pick, while Red may have speed advantages.

## Implications for Data Analysis

### Win Rate Above Expectation (WRAE)
The auction system is successful if the **WRAE is 0.00**. 
- If a town has a **Positive WRAE** (like Fortress or Tower), it means the town is **Underpriced**: the advantage it provides is greater than the gold players currently bid/pay for it.
- If a town has a **Negative WRAE** (like Necropolis), it means the town is **Overpriced**: players are paying a premium that exceeds the actual win-probability gain the town provides at the elite level.

### Future Research: Fair Price Models
Using the `handicap` data, we can build models to predict the exact "equilibrium point" (in gold) for any town-town matchup where the win probability would be exactly 50%.

> [!TIP]
> This "Economy of Balance" is why simple win rates are misleading. A 55% win rate for Fortress might look like a balance issue, but it's actually an **efficiency issue** in the current auction meta.
