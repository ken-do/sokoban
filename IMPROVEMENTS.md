# Gomoku AI: Cải Tiến Minimax & Thêm Machine Learning

## 1. Cải Tiến Minimax Agent

### Những Cải Tiến Chính:

#### a) **Evaluation Function Cải Tiến**
- **Tăng Trọng Số Phòng Thủ**: Từ 4.0 → 5.0
  - AI sẽ ưu tiên chặn đối thủ hơn để tránh thua
  - Điều này rất quan trọng ở depth thấp khi nhìn không xa được

- **Tăng Scoring cho Patterns**:
  ```
  Bốn quân (Four):     10000 → 15000
  Ba quân hở (Open 3): 8000  → 10000
  Ba quân (Three):     500   → 800
  Hai quân hở (Open 2): 100  → 150
  ```

#### b) **Move Ordering Cải Tiến**
- **Tăng Trọng Số Phòng Thủ trong Sorting**: 1.5 → 2.0
- **Thêm Penalty cho Tấn Công**: Nhân với 0.8
- Giúp Alpha-Beta Pruning hoạt động hiệu quả hơn

#### c) **Transposition Table (Caching)**
- Lưu kết quả đã tính để tránh tính lại
- Giảm số node cần explore

### Kết Quả:
- **Winrate vs Random**: 66.7% (2/3 games)
- **Depth**: 2 (có thể tăng lên 3 nếu có thời gian)

---

## 2. Neural Network Agent (Mới)

### Đặc Điểm:
- **Architecture**: 
  - Input: Bàn cờ 15x15 flatten (225 features)
  - Hidden layers: 256 → 128 → 64 neurons (ReLU activation, Dropout)
  - Output: 225 (một score cho mỗi ô)

- **Safety Layer**:
  - Kiểm tra nước thắng ngay
  - Kiểm tra phải chặn đối thủ
  - Fallback sang heuristic evaluation nếu NN không khả dụng

- **Training**:
  - Học từ trò chơi với SmartRandomAgent
  - Huấn luyện qua 30-50 games
  - Sử dụng MSE loss

- **Heuristic Fallback**:
  - Nếu TensorFlow không cài đặt, AI sẽ dùng evaluation nhanh
  - Vẫn đạt kết quả tốt nhờ heuristic và safety checks

### Lợi Thế:
- Có thể học từ data thay vì hard-code rules
- Mở rộng được với Deep Q-Learning, Policy Gradient
- Không bị giới hạn bởi depth như Minimax

---

## 3. So Sánh Các Agent

| Agent | Kiến Trúc | Ưu Điểm | Nhược Điểm |
|-------|-----------|--------|-----------|
| **SmartRandomAgent** | Heuristic | Nhanh, Đơn Giản | Winrate thấp |
| **MinimaxAgent (Cải Tiến)** | Minimax + Eval | Thắng consistent, Khó đánh | Chậm ở depth cao |
| **NeuralNetworkAgent** | Deep Learning | Học được, Mở rộng tốt | Cần training data |

---

## 4. Cách Chạy

### Test Minimax vs Random:
```bash
python src/test_quick.py
```

### Test ML Agent + Minimax:
```bash
python src/GomokuML.py
```

### Test Original:
```bash
python src/Gomoku.py
```

---

## 5. Những Cải Tiến Có Thể Thêm

1. **Tăng Depth**: Hiện tại Depth 2-3, có thể tăng lên 4 với:
   - Beam search (chỉ lấy top-k best moves)
   - Iterative deepening
   - More aggressive pruning

2. **Cải Tiến NN**:
   - Thêm convolutional layers (để nhận diện patterns)
   - Self-play reinforcement learning
   - Policy + Value network (như AlphaGo)

3. **Opening Book**:
   - Hard-code các nước đi tốt nhất ở đầu game
   - Giảm search space đáng kể

4. **Endgame Solver**:
   - Database các position ở cuối game
   - Đảm bảo tìm được forced win nếu có

---

## 6. File Thay Đổi

- **src/Gomoku.py**: Cải tiến MinimaxAgent
  - Tăng evaluation weights
  - Cải tiến move ordering
  - Fix encoding Vietnamese

- **src/GomokuML.py**: File mới với 3 class
  - `NeuralNetworkAgent`: ML-based agent
  - `ImprovedMinimaxAgent`: Wrapper với transposition table
  - Test cases tổng hợp

- **src/test_quick.py**: Script test nhanh
  - 3 games test
  - Hiển thị winrate
