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
from multiprocessing import Process, Queue, Pool
from functools import partial

def run_single_old_game(game_num):
    """Run a single game for old version"""
    minimax = MinimaxAgentOldVersion(name="OldMinimax", max_depth=3, search_radius=2)
    random_agent = SmartRandomAgentOldVersion(name="Random")
    winner, moves = play_game_old(minimax, random_agent, display=False)
    return (game_num, winner, moves)

def run_single_new_game(game_num):
    """Run a single game for new version"""
    minimax = MinimaxAgent(name="NewMinimax", max_depth=3, search_radius=2)
    random_agent = SmartRandomAgent(name="Random")
    winner, moves = play_game_new(minimax, random_agent, display=False)
    return (game_num, winner, moves)

def run_single_ml_game(game_num):
    """Run a single game for ML version"""
    ml_agent = SimpleNNAgent(name="ML_Agent")
    random_agent = SmartRandomAgent(name="Random")
    winner, moves = play_game_new(ml_agent, random_agent, display=False)
    return (game_num, winner, moves)

def benchmark_old_vs_random(queue=None):
    """Test Old Minimax Agent vs Random"""
    print("\n" + "=" * 80)
    print("OLD MINIMAX AGENT vs RANDOM (30 games) - PARALLEL")
    print("=" * 80 + "\n")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    # Run games in parallel
    with Pool(processes=6) as pool:
        results = pool.map(run_single_old_game, range(30))
    
    # Process results
    for game_num, winner, moves in sorted(results, key=lambda x: x[0]):
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"  [OLD] Game {game_num+1}/30... {result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 30
    print(f"\n  [OLD] Result: {wins}/30 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  [OLD] Time: {elapsed:.1f}s | Avg: {elapsed/30:.2f}s/game\n")
    
    if queue:
        queue.put(("old", winrate, wins, draws, losses, elapsed))
    return winrate

def benchmark_new_vs_random(queue=None):
    """Test New Minimax Agent vs Random"""
    print("\n" + "=" * 80)
    print("NEW MINIMAX AGENT vs RANDOM (30 games) - PARALLEL")
    print("=" * 80 + "\n")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    # Run games in parallel
    with Pool(processes=6) as pool:
        results = pool.map(run_single_new_game, range(30))
    
    # Process results
    for game_num, winner, moves in sorted(results, key=lambda x: x[0]):
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"  [NEW] Game {game_num+1}/30... {result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 30
    print(f"\n  [NEW] Result: {wins}/30 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  [NEW] Time: {elapsed:.1f}s | Avg: {elapsed/30:.2f}s/game\n")
    
    if queue:
        queue.put(("new", winrate, wins, draws, losses, elapsed))
    return winrate

def benchmark_ml_vs_random(queue=None):
    """Test ML Agent vs Random"""
    print("\n" + "=" * 80)
    print("ML AGENT vs RANDOM (20 games) - PARALLEL")
    print("=" * 80 + "\n")
    
    wins = 0
    losses = 0
    draws = 0
    start = time.time()
    
    # Run games in parallel
    with Pool(processes=6) as pool:
        results = pool.map(run_single_ml_game, range(20))
    
    # Process results
    for game_num, winner, moves in sorted(results, key=lambda x: x[0]):
        if winner == 1:
            wins += 1
            result = "WIN"
        elif winner == 0:
            draws += 1
            result = "DRAW"
        else:
            losses += 1
            result = "LOSS"
        print(f"  [ML] Game {game_num+1}/20... {result} ({moves} moves)")
    
    elapsed = time.time() - start
    winrate = 100 * wins / 20
    print(f"\n  [ML] Result: {wins}/20 = {winrate:.1f}% | Draws: {draws} | Losses: {losses}")
    print(f"  [ML] Time: {elapsed:.1f}s | Avg: {elapsed/20:.2f}s/game\n")
    
    if queue:
        queue.put(("ml", winrate, wins, draws, losses, elapsed))
    return winrate

def main():
    print("\n" + "=" * 80)
    print("GOMOKU AGENTS COMPARISON TEST (PARALLEL EXECUTION)")
    print("=" * 80)
    
    # Create queue for collecting results
    result_queue = Queue()
    
    # Create processes for parallel execution
    p1 = Process(target=benchmark_old_vs_random, args=(result_queue,))
    p2 = Process(target=benchmark_new_vs_random, args=(result_queue,))
    p3 = Process(target=benchmark_ml_vs_random, args=(result_queue,))
    
    # Start all processes
    start_time = time.time()
    p1.start()
    p2.start()
    p3.start()
    
    # Wait for all to complete
    p1.join()
    p2.join()
    p3.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results = {}
    while not result_queue.empty():
        name, winrate, wins, draws, losses, elapsed = result_queue.get()
        results[name] = {
            'winrate': winrate,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'time': elapsed
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"Old Minimax: {results.get('old', {}).get('winrate', 0):.1f}%")
    print(f"New Minimax: {results.get('new', {}).get('winrate', 0):.1f}%")
    print(f"ML Agent:    {results.get('ml', {}).get('winrate', 0):.1f}%")
    print(f"\nTotal parallel execution time: {total_time:.1f}s")
    print("=" * 80 + "\n")
    
    old_winrate = results.get('old', {}).get('winrate', 0)
    new_winrate = results.get('new', {}).get('winrate', 0)
    ml_winrate = results.get('ml', {}).get('winrate', 0)
    
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
