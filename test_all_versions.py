#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test comparing Old Gomoku vs New Gomoku vs ML implementations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import old version
from Gomoku_old import (
    GomokuGameOldVersion, 
    MinimaxAgentOldVersion, 
    SmartRandomAgentOldVersion,
    play_game as play_game_old
)

# Import new versions
from Gomoku import GomokuGame, MinimaxAgent, SmartRandomAgent, play_game as play_game_new
from GomokuML import SimpleNNAgent
import time

def benchmark_old_vs_random():
    """Test Old Minimax Agent vs Random"""
    print("\n" + "=" * 80)
    print("OLD MINIMAX AGENT vs RANDOM (30 games)")
    print("=" * 80 + "\n")
    
    minimax = MinimaxAgentOldVersion(name="OldMinimax", max_depth=3, search_radius=2)
    random_agent = SmartRandomAgentOldVersion(name="Random")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    for i in range(30):
        print(f"  Game {i+1}/30...", end=" ", flush=True)
        winner, moves = play_game_old(minimax, random_agent, display=False)
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"{result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 30
    print(f"\n  Result: {wins}/30 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  Time: {elapsed:.1f}s | Avg: {elapsed/30:.2f}s/game\n")
    
    return winrate

def benchmark_new_vs_random():
    """Test New Minimax Agent vs Random"""
    print("\n" + "=" * 80)
    print("NEW MINIMAX AGENT vs RANDOM (30 games)")
    print("=" * 80 + "\n")
    
    minimax = MinimaxAgent(name="NewMinimax", max_depth=3, search_radius=2)
    random_agent = SmartRandomAgent(name="Random")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    for i in range(30):
        print(f"  Game {i+1}/30...", end=" ", flush=True)
        winner, moves = play_game_new(minimax, random_agent, display=False)
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"{result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 30
    print(f"\n  Result: {wins}/30 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  Time: {elapsed:.1f}s | Avg: {elapsed/30:.2f}s/game\n")
    
    return winrate

def benchmark_ml_vs_random():
    """Test ML Agent vs Random"""
    print("\n" + "=" * 80)
    print("ML AGENT vs RANDOM (20 games)")
    print("=" * 80 + "\n")
    
    ml_agent = SimpleNNAgent(name="ML_Agent")
    random_agent = SmartRandomAgent(name="Random")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    for i in range(20):
        print(f"  Game {i+1}/20...", end=" ", flush=True)
        winner, moves = play_game_new(ml_agent, random_agent, display=False)
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"{result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 20
    print(f"\n  Result: {wins}/20 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  Time: {elapsed:.1f}s | Avg: {elapsed/20:.2f}s/game\n")
    
    return winrate

def main():
    print("\n" + "=" * 80)
    print("GOMOKU AGENTS COMPARISON TEST")
    print("=" * 80)
    
    old_winrate = benchmark_old_vs_random()
    new_winrate = benchmark_new_vs_random()
    ml_winrate = benchmark_ml_vs_random()
    
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"Old Minimax: {old_winrate:.1f}%")
    print(f"New Minimax: {new_winrate:.1f}%")
    print(f"ML Agent:    {ml_winrate:.1f}%")
    print("=" * 80 + "\n")
    
    improvement = new_winrate - old_winrate
    print(f"Improvement from Old to New Minimax: {improvement:+.1f}%")
    
    if new_winrate >= 90:
        print("✓ New Minimax meets 90% requirement!")
    else:
        print(f"✗ New Minimax is {90 - new_winrate:.1f}% below target")
    
    if ml_winrate >= 60:
        print("✓ ML Agent meets 60% requirement!")
    else:
        print(f"✗ ML Agent is {60 - ml_winrate:.1f}% below target")

if __name__ == "__main__":
    main()
