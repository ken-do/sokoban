#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Agent for Gomoku - Neural Network
Yêu cầu: Đạt 60% winrate vs Random Agent
"""

import numpy as np
from typing import Tuple, List
import random

class SimpleNNAgent:
    """
    Simple Neural Network Agent cho Gomoku
    - Input: Board state (15x15 = 225 features)
    - Hidden: 128 neurons
    - Output: Move evaluation scores
    - Training: Q-learning with board patterns
    """
    
    def __init__(self, name="NeuralNetwork_60%", board_size=15):
        self.name = name
        self.board_size = board_size
        self.learning_rate = 0.01
        
        # Simple neural network weights
        self.w1 = np.random.randn(225, 128) * 0.01  # Input -> Hidden
        self.b1 = np.zeros((1, 128))
        self.w2 = np.random.randn(128, 1) * 0.01    # Hidden -> Output
        self.b2 = np.zeros((1, 1))
        
        # Pattern database for heuristic evaluation
        self.pattern_weights = self._init_patterns()

    def _init_patterns(self):
        """Initialize pattern recognition weights"""
        return {
            'open_four': 100,
            'four': 50,
            'open_three': 40,
            'three': 10,
            'open_two': 5,
        }

    def _relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)

    def _sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _forward(self, x):
        """Forward pass"""
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self._sigmoid(z2)
        return a2

    def get_move(self, game) -> Tuple[int, int]:
        """
        Tìm nước đi tốt nhất bằng:
        1. Pattern-based heuristic (tấn công/phòng thủ)
        2. NN evaluation (học từ board state)
        3. Random selection từ top moves
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # First move - center
        if len(game.move_history) == 0:
            c = game.board_size // 2
            return (c, c)

        my_player = game.current_player
        opponent = 3 - my_player

        # 1. Check immediate WIN
        for move in valid_moves:
            r, c = move
            game.board[r][c] = my_player
            if game.check_winner(r, c):
                game.board[r][c] = 0
                return (r, c)
            game.board[r][c] = 0

        # 2. Check immediate BLOCK
        for move in valid_moves:
            r, c = move
            game.board[r][c] = opponent
            if game.check_winner(r, c):
                game.board[r][c] = 0
                return (r, c)
            game.board[r][c] = 0

        # 3. Pattern-based scoring
        scored_moves = []
        for move in valid_moves:
            r, c = move
            
            # Heuristic score
            h_score = self._heuristic_eval(game, r, c, my_player, opponent)
            
            # NN score
            board_flat = game.board.flatten().reshape(1, -1)
            nn_score = self._forward(board_flat)[0][0]
            
            # Combined score
            total_score = h_score * 0.7 + nn_score * 0.3
            
            # Add randomness
            total_score += random.random() * 10
            
            scored_moves.append((move, total_score))

        # Pick top move
        best_move = max(scored_moves, key=lambda x: x[1])[0]
        return best_move

    def _heuristic_eval(self, game, row, col, my_player, opponent):
        """Pattern-based heuristic evaluation"""
        score = 0
        
        # Evaluate attack
        game.board[row][col] = my_player
        attack_score = self._count_patterns(game, row, col, my_player)
        score += attack_score * 1.2
        
        # Evaluate defense
        game.board[row][col] = opponent
        defense_score = self._count_patterns(game, row, col, opponent)
        score += defense_score * 1.0
        
        game.board[row][col] = 0
        
        return score

    def _count_patterns(self, game, row, col, player):
        """Count patterns around a position"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            open_ends = 0
            
            # Count in both directions
            for d in [1, -1]:
                r, c = row + dx * d, col + dy * d
                while 0 <= r < game.board_size and 0 <= c < game.board_size:
                    if game.board[r][c] == player:
                        count += 1
                        r += dx * d
                        c += dy * d
                    elif game.board[r][c] == 0:
                        open_ends += 1
                        break
                    else:
                        break
            
            # Pattern scoring
            if count >= 5:
                score += 1000
            elif count == 4:
                score += 500 if open_ends > 0 else 100
            elif count == 3:
                score += 100 if open_ends == 2 else 20
            elif count == 2:
                score += 10 if open_ends == 2 else 0
        
        return score

    def train(self, board, move, result):
        """
        Simple training update (Q-learning style)
        result: 1 if win, 0 if draw, -1 if loss
        """
        board_flat = board.flatten().reshape(1, -1)
        
        # Forward pass
        z1 = np.dot(board_flat, self.w1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        prediction = self._sigmoid(z2)[0][0]
        
        # Target: win=1, draw=0.5, loss=0
        target = max(0, min(1, (result + 1) / 2))
        
        # Gradient descent
        dz2 = prediction - target
        dw2 = np.dot(a1.T, dz2.reshape(1, 1))
        db2 = dz2.reshape(1, 1)
        
        dz1 = np.dot(dz2.reshape(1, 1), self.w2.T) * (a1 > 0)
        dw1 = np.dot(board_flat.T, dz1)
        db1 = dz1
        
        # Update weights
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
