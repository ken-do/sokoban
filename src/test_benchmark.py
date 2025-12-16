#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick benchmark: So sánh Minimax depths
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Gomoku import GomokuGame, SmartRandomAgent, MinimaxAgent, play_game
import time

def benchmark():
    print("\n" + "=" * 80)
    print("MINIMAX DEPTH BENCHMARK (vs Random)")
    print("=" * 80 + "\n")
    
    configs = [
        (3, 2, 20),  # depth, search_radius, num_games
    ]
    
    for depth, radius, num_games in configs:
        print(f"Testing Depth {depth}, Radius {radius} ({num_games} games)")
        print("-" * 80)
        
        agent = MinimaxAgent(name=f"D{depth}R{radius}", max_depth=depth, search_radius=radius)
        random = SmartRandomAgent(name="Random")
        
        wins = 0
        losses = 0
        draws = 0
        start = time.time()
        
        for i in range(num_games):
            print(f"  Game {i+1}/{num_games}...", end=" ", flush=True)
            winner, moves = play_game(agent, random, display=False)
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
        print(f"\n  Result: {wins}/{num_games} = {100*wins/num_games:.0f}% | Draws: {draws} | Losses: {losses}")
        print(f"  Time: {elapsed:.1f}s | Avg: {elapsed/num_games:.2f}s/game\n")

if __name__ == "__main__":
    benchmark()
