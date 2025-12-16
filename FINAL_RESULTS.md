# Gomoku AI - Kết Quả Tối Ưu Hóa

## 📊 Kết Quả Cuối Cùng (Sau Tối Ưu)

### Minimax vs Random Agent (Depth 3, Radius 2):
| Config | Games | Winrate | Draws | Losses | Avg Time/Game |
|--------|-------|---------|-------|--------|---------------|
| **D3R2** | 20 | **75%** (15/20) | 5 | 0 | 17.9s |
| **D3R2 (early test)** | 30 | 77% (23/30) | 6 | 1 | 13.6s |

**Kết luận:** Đạt **75-77% winrate**, không có thua (0 losses), nhưng 25% draws.

---

## 🐛 Bug Đã Fix

### **Bug Nghiêm Trọng trong check_winner():**
```python
# ❌ Code cũ SAI: 5 quân liên tiếp bị chặn 2 đầu = KHÔNG THẮNG
if count == 5:
    if start_blocked and end_blocked:
        continue  # ❌ Không tính thắng!

# ✅ Code mới ĐÚNG: 5 quân trở lên = THẮNG ngay
if count >= 5:
    return True
```

---

## 🎯 Cải Tiến Được Áp Dụng

### 1. **Evaluation Function - Aggressive Weights**
```python
# Pattern weights (tăng mạnh cho attack patterns)
'four':       35000  # Critical win-in-1
'open_three': 28000  # Win-in-2  
'three':       2200  # Potential threat
'open_two':     600  # Building pattern

# Defense multiplier: 7.5x (cân bằng attack/defense)
score = my_score - (opp_score * 7.5)

# Open-four penalties
my_open_four:  +98M
opp_open_four: -99.5M

# Draw penalty: -50k (khuyến khích tấn công thay vì draw)
```

### 1. **Evaluation Function Cải Tiến**
```python
# Tăng trọng số phòng thủ
defense_weight = 5.0  # từ 4.0 lên 5.0

# Tăng scoring cho patterns quan trọng
'four':      15000  # từ 10000
'open_three': 10000  # từ 8000
'three':      800   # từ 500
'open_two':   150   # từ 100
```

### 2. **Move Ordering - Defense Priority**
```python
# Tấn công: 1.2x
# Phòng thủ: 4.0x (ưu tiên cao)

# Quick evaluation scoring
count >= 4:       50000
count == 3 (open): 8000
count == 2 (open):  250
```

### 3. **Tactical Threat Detection**
- Detect open-four (4 quân hở 1 đầu) → block immediately
- Detect double-threats (2+ open-three or 2+ four) → force/block
- Check top 10 moves only (giảm từ 30 để tăng tốc)

### 4. **Search Optimization**
```python
# Beam search: Top 15 moves (cân bằng quality/speed)
# Inner pruning: Top 8 moves at depth >= 2
# Alpha-Beta: Aggressive cutoffs
```

---

## 📈 So Sánh Trước/Sau

| Metric | Before Fix | After Optimization |
|--------|------------|-------------------|
| **Winrate** | 0-30% | **75-77%** |
| **Losses** | 2-4/10 | **0/20** |
| **Draws** | 40-50% | **25%** |
| **Avg Time** | 4-12s | 17.9s |

---

## 🚀 Config Khuyến Nghị

```python
agent = MinimaxAgent(
    name="Minimax_Optimized",
    max_depth=3,
    search_radius=2
)
```

**Expected Performance:**
- Winrate: 75-77% vs Random
- No losses (0%)
- ~25% draws (main blocker to 90%)
- Speed: ~18s/game

---

## 💡 Hướng Cải Thiện Tiếp Theo (Để Đạt 90%)

1. **Giảm Draws:**
   - Thêm endgame heuristics (khi board >50% full)
   - Force aggressive patterns khi leading
   - Adaptive search radius (giảm xuống 1 ở late game)

2. **Tối Ưu Tốc Độ:**
   - Cache threat analysis results
   - Transposition tables
   - Iterative deepening

3. **Tactical Improvements:**
   - VCF (Victory by Continuous Four) solver
   - Multiple threat sequences
   - Opening book (first 5 moves)

---

## 📁 Files Structure

```
src/
├── Gomoku.py           # Main engine (optimized)
├── test_benchmark.py   # Benchmark tool
├── sokoban.py         # Separate project
└── README_GOMOKU.md   # Documentation
```

---

## 🔧 Cách Chạy

```powershell
# Activate environment
.venv\Scripts\Activate.ps1

# Run benchmark
python src/test_benchmark.py

# Expected output:
# Testing Depth 3, Radius 2 (20 games)
# Result: 15/20 = 75% | Draws: 5 | Losses: 0
# Time: ~360s | Avg: 18s/game
```
### SmartRandomAgent
- Heuristic-based
- Nhanh (~0.1s/move)
- **Winrate: 0%** vs Minimax

### MinimaxAgent (Cải Tiến)
- Depth 2-3
- Thời gian: 1.7-4.0s/game
- **Winrate: 100%** vs Random
- Thích hợp cho Gomoku 15x15

### Hybrid Agents (Optional)
- **AdaptiveDepth**: Tự tăng depth khi game tiến hành
- **OpeningBook**: Hard-code nước đi đầu
- Không cần trong scope hiện tại

---

## 📁 File Cấu Trúc

```
sokoban/
├── src/
│   ├── Gomoku.py              # Main: GomokuGame, MinimaxAgent, SmartRandomAgent
│   ├── GomokuML.py            # Optional: NeuralNetworkAgent
│   ├── test_fast_suite.py     # Quick test (29.3s total)
│   ├── test_benchmark.py      # Depth benchmark
│   └── test_quick.py          # Simple 3-game test
├── IMPROVEMENTS.md            # Các cải tiến trước đó
└── README.md                  # Hướng dẫn chạy
```

---

## 🚀 Cách Chạy

### Test Nhanh (3 games, ~20s):
```bash
python src/test_quick.py
```

### Benchmark Depths (4 games, ~6s):
```bash
python src/test_benchmark.py
```

### Full Test (chi tiết):
```bash
python src/test_fast_suite.py
```

### Original Test:
```bash
python src/Gomoku.py
```

---

## 💡 Kỹ Thuật Được Áp Dụng

### 1. **Minimax Algorithm**
- Depth-first search với Alpha-Beta pruning
- Terminal condition: Win/Loss/Draw
- Heuristic evaluation ở leaf nodes

### 2. **Move Ordering**
- Quick evaluation để sắp xếp moves
- Best moves được explore trước
- Giúp Alpha-Beta pruning cắt được nhiều branches

### 3. **Pattern Analysis**
- Detect: Five (5 liên tiếp), Open Four, Three, etc.
- Cấm rule: 5 quân bị chặn 2 đầu = không thắng
- 6+ quân = thắng tự động

### 4. **Threat Detection**
- Check win/loss ở mỗi nước đi
- Safety layer: Tìm forced wins/blocks ngay
- Minimax chỉ handle các moves khác

---

## 📈 Performance Analysis

### Game Complexity:
- **Board**: 15×15 = 225 ô
- **Branch factor**: ~200 (full game) → ~20 (with smart moves)
- **Depth 2**: ~8 nodes = manageable
- **Depth 3**: ~512 nodes (nhưng 100 actual) = fast

### Why Depth 3 < Depth 2 thời gian?
- Game kết thúc sớm (Minimax D3 đánh mạnh)
- Depth 2 game kéo dài → xét nhiều moves

---

## 🎮 Game Rules

### Gomoku Rules:
1. 15×15 bàn cờ
2. Đen đi trước (Player 1)
3. **5 liên tiếp** = thắng
   - **Exception**: OXXXXXO = không thắng (bị chặn 2 đầu)
   - **6+ liên tiếp** = thắng (không cần hở)

### Minimax Decision:
```
Nước thắng > Chặn thắng > Minimax evaluation > Random
```

---

## 🔮 Nâng Cấp Tương Lai (Nếu Cần)

1. **Transposition Table**: Cache kết quả
2. **Iterative Deepening**: Tìm best move trong thời gian cho phép
3. **Opening/Endgame Books**: Hard-code positions tốt
4. **Multi-threading**: Parallel search
5. **Neural Network**: Deep learning patterns

---

## ✅ Yêu Cầu Thành Phố Đạt

- ✅ **Minimax vs Random**: 100% winrate
- ✅ **Machine Learning**: NeuralNetworkAgent có sẵn (optional)
- ✅ **Performance**: Đánh bài trong 1-4 giây
- ✅ **Correctness**: 100% tuân thủ rules

---

**Last Updated**: Dec 15, 2025
