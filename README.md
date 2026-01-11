# Sokoban & Gomoku Games

This repository contains implementations of two classic games:
1. **Sokoban** - A puzzle game solver with multiple search algorithms (DFS, BFS, A*)
2. **Gomoku** - A strategy board game with AI agents including machine learning-based players

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone or download this repository
2. Navigate to the project directory:
```bash
cd sokoban
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## How to Run

### Sokoban Puzzle Solver

Run the Sokoban solver:
```bash
python src/sokoban.py
```

The solver supports multiple search algorithms:
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- A* Search Algorithm
- Hill Climbing

### Gomoku Game

Run the basic Gomoku game:
```bash
python src/Gomoku.py
```

Run the Gomoku game with ML-enhanced agent:
```bash
python src/GomokuML.py
```

## Project Structure

```
sokoban/
├── src/
│   ├── sokoban.py      # Sokoban puzzle solver
│   ├── Gomoku.py       # Gomoku game implementation
│   └── GomokuML.py     # ML-enhanced Gomoku agent
├── test-cases/
│   ├── cosmos-test-cases.txt
│   └── microban-test-cases.txt
├── docs/
│   ├── requirements.md
│   ├── sample-memory-comparison.md
│   └── sample-time-comparison.md
├── requirements.txt
├── setup.py
└── README.md
```

## References

1. https://en.wikipedia.org/wiki/Sokoban
2. http://en.wikipedia.org/wiki/Depth-first_search
3. http://en.wikipedia.org/wiki/Breadth-first_search
4. http://en.wikipedia.org/wiki/Hill_climbing
5. https://en.wikipedia.org/wiki/A*_search_algorithm
6. https://ksokoban.online/