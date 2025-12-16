# Gomoku AI - Minimax Implementation

## Files
- `src/Gomoku.py` - Main game engine with optimized Minimax agent
- `src/test_benchmark.py` - Benchmark tool
- `src/sokoban.py` - Sokoban game (separate project)

## Quick Start

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Run benchmark
python src/test_benchmark.py
```

## Minimax Configuration

**Recommended settings:**
- **Depth**: 3
- **Search Radius**: 2
- **Expected Winrate**: 85-90% vs Random

## Key Features

### Evaluation Function
- Pattern scoring: Four (30k), Open-three (22k), Three (1.5k), Open-two (400)
- Defense multiplier: 7.0x
- Threat-first selection for tactical moves

### Optimizations
- Alpha-Beta pruning
- Beam search (top 15 moves)
- Smart move ordering (defense priority 4.0x)
- Forced threat/block detection

### Performance
- Avg time: ~10-15s per game at depth 3
- Search space reduction via radius-based candidates

## Tuning

Adjust in `Gomoku.py`:
- `evaluate_board()`: Pattern weights & defense multiplier
- `get_move()`: Beam width (currently 15)
- `_minimax()`: Inner branch factor (currently 8)
- `_sort_moves_by_priority()`: Attack/defense balance
