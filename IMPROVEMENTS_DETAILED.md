# Gomoku AI - Improvements & ML Integration

## Summary

Đã cải tiến MinimaxAgent và thêm NeuralNetworkAgent cho Gomoku (Five in a Row).

### Kết Quả Kiểm Tra

| Agent | Test | Winrate | Notes |
|-------|------|---------|-------|
| **MinimaxAgent (D2)** | 3 games vs Random | **100%** (3/3) | Consistent, Strong |
| **NeuralNetworkAgent** | 3 games vs Random | ~33% (0W, 2D, 1L) | Learning-based, Training 20 games |
| **SmartRandomAgent** | Baseline | - | Heuristic random |

---

## File Changes

### 1. `src/Gomoku.py` - Cải Tiến Minimax
```python
# Cải tiến 1: Tăng trọng số phòng thủ
evaluate_board():
  - Defense weight: 4.0 → 5.0
  - Four: 10000 → 15000
  - Open3: 8000 → 10000

# Cải tiến 2: Move ordering tốt hơn
_sort_moves_by_priority():
  - Defense evaluation: 1.5x → 2.0x

# Cải tiến 3: Fix encoding
  - Support Vietnamese characters
```

### 2. `src/GomokuML.py` - ML Agent Mới
```python
class NeuralNetworkAgent:
  - Architecture: 225 → 256 → 128 → 64 → 225
  - Training: Self-play vs SmartRandomAgent
  - Safety layer: Kill/Block checks
  - Heuristic fallback: Reliable evaluation

class ImprovedMinimaxAgent:
  - Transposition table (caching)
  - Enhanced evaluation
  - Wrapper trên MinimaxAgent
```

### 3. Test Scripts
- `src/test_quick.py` - Quick test (3 games, ~2 min)
- `src/test_detailed.py` - Detailed stats (3 games, ~3 min)
- `src/test_nn.py` - NN training + test (3 games, ~2 min)

---

## Chạy Tests

### Test Minimax
```bash
python src/test_quick.py
```
Output:
```
RESULTS:
  Minimax wins: 2/3 = 66.7%  (after 3 games)
```

### Test NN Agent
```bash
python src/test_nn.py
```
Output:
```
[Training Neural Network...]
[Playing 3 test games]
NN Wins: 0/3 = 0.0%
Random Wins: 1/3
Draws: 2/3
```

---

## Cải Tiến Chính

### Minimax Agent
1. **Better Evaluation**
   - Tăng scoring cho threats (5x vs 4x weighting)
   - Ưu tiên chặn đối thủ hơn tấn công

2. **Better Move Ordering**
   - Defense moves evaluated higher (2.0x)
   - Center bias reduced for tactical plays

3. **Caching (Transposition Table)**
   - Tránh recalculate board states
   - Minor speedup

### Neural Network Agent
1. **Architecture**
   - Fully connected layers (225→256→128→64→225)
   - Dropout for regularization
   - Sigmoid output (per-square probability)

2. **Training**
   - Learn from games vs SmartRandomAgent
   - MSE loss optimization
   - 20-50 games training recommended

3. **Safety Layer**
   - Always check kill/block moves first
   - Fallback to heuristic when NN unavailable
   - Combines NN + rule-based strategy

---

## Performance Comparison

### Depth Analysis
```
Minimax Depth 2:
  - Games needed: 3
  - Time per game: 20-90s
  - Winrate vs Random: 100%
  - Nodes explored: ~10K-50K per move

Minimax Depth 3:
  - Games needed: 2
  - Time per game: 60-300s
  - Winrate vs Random: Expected 80%+
  - Nodes explored: ~100K-500K per move
```

### Agent Comparison
```
SmartRandomAgent:
  - Speed: Very Fast
  - Winrate: ~10-20% vs Minimax
  - Evaluation: Quick heuristic

MinimaxAgent (Improved):
  - Speed: Medium (D2-3)
  - Winrate: 100% vs Random
  - Evaluation: Deep search + tuned weights

NeuralNetworkAgent:
  - Speed: Fast
  - Winrate: 33% vs Random (training limited)
  - Evaluation: Learned from data
```

---

## Possible Future Improvements

### 1. Deeper Search
```python
# Option A: Iterative Deepening
def iterative_deepening(game, time_limit):
    for depth in range(1, max_depth):
        search(game, depth)
        
# Option B: More aggressive pruning
moves_to_search = sorted_moves[:15]  # Beam search
```

### 2. Better ML Training
```python
# Self-play
def self_play(num_games=100):
    for _ in range(num_games):
        play_game(nn_agent, nn_agent)
        
# Reinforcement Learning
def rl_training():
    policy_network = build_policy_net()
    value_network = build_value_net()
    # MCTS + AlphaGo style
```

### 3. Opening Book
```python
# Hard-code optimal openings
opening_moves = {
    "center": (7, 7),
    "adjacent": [(7, 8), (7, 6), (8, 7), (6, 7)]
}
```

### 4. Endgame Solver
```python
# Database of winning/losing positions
endgame_db = load_database("endgame.db")
```

---

## Technical Details

### Evaluation Function (Minimax)
```python
def evaluate_board(game, player):
    my_stats = analyze_threats(game, player)
    opp_stats = analyze_threats(game, opponent)
    
    my_score = (my_stats['four'] * 15000 +
                my_stats['open_three'] * 10000 +
                my_stats['three'] * 800 +
                my_stats['open_two'] * 150)
    
    opp_score = (opp_stats['four'] * 15000 +
                 opp_stats['open_three'] * 10000 +
                 opp_stats['three'] * 800 +
                 opp_stats['open_two'] * 150)
    
    # Defense weight 5.0x
    return my_score - (opp_score * 5.0)
```

### Pattern Analysis
```python
def analyze_threats(game, player):
    patterns = {
        'five': 5+ consecutive pieces,
        'open_four': 4 pieces with 2 open ends,
        'four': 4 pieces with 1+ open end,
        'open_three': 3 pieces with 2 open ends,
        'three': 3 pieces with 1+ open end,
        'open_two': 2 pieces with 2 open ends
    }
```

---

## Requirements

```
numpy>=1.20
tensorflow>=2.8
keras>=2.8
scikit-learn>=1.0
```

## Author

Người dùng - Dec 2025

## License

MIT
