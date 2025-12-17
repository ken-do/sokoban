# Gomoku ML Agent - Kịch Bản Thuyết Trình (5 Phút)

## Slide 1: Giới Thiệu
**[Hiển thị cell tiêu đề]**

"Xin chào mọi người. Hôm nay tôi sẽ trình bày về Gomoku ML Agent - một hệ thống heuristic chơi cờ caro 5 ô liên tiếp.

Mục tiêu: đạt ≥60% tỷ lệ thắng so với đối thủ ngẫu nhiên thông minh. Đặc biệt, chúng ta KHÔNG dùng neural network, mà dùng pattern recognition cổ điển.

**Tính năng chính:**
- Pattern scoring không cần neural network
- Phát hiện nước thắng/chặn ngay lập tức
- Phân tích 4 hướng
- Cân bằng tấn công-phòng thủ tối ưu"

---

## Slide 2-4: Setup Nhanh
**[Chạy cell 3 cài đặt]**

"Cell này cài đặt các thư viện cần thiết: numpy, pandas, matplotlib, psutil.

**[Chạy cell 4 import]**

Import thành công. Chúng ta đã sẵn sàng!"

---

## Slide 5-8: Chiến Lược 3 Tầng
**[Hiển thị pseudocode]**

"Agent sử dụng quyết định 3 tầng:

**Tầng 1:** Nước đầu → giữa bàn cờ
**Tầng 2:** Kiểm tra thắng ngay → đánh luôn
**Tầng 3:** Kiểm tra đối thủ sắp thắng → chặn ngay
**Tầng 4:** Đánh giá tất cả nước còn lại bằng pattern scoring

**[Run cell 8]**

Công thức: `điểm = tấn_công × 4.5 + phòng_thủ × 1.5`

Tỷ lệ 3:1 ưu tiên tấn công nhưng không bỏ sót phòng thủ quan trọng."

---

## Slide 9-10: Pattern Scoring
**[Hiển thị bảng điểm]**

"Trái tim của thuật toán - bảng điểm các pattern:

**[Run cell 10]**

- 5 liên tiếp: 3 triệu điểm → thắng luôn!
- Open-4 (2 đầu mở): 300,000 điểm → mối đe dọa nghiêm trọng
- Open-3: 50,000 điểm → 10 lần mạnh hơn half-open-3
- Open-2: 2,000 điểm → xây dựng thế

**Insight quan trọng:** Pattern có 2 đầu mở mạnh GẤP BỘI pattern bị chặn!"

---

## Slide 11-12: Thuật Toán Đánh Giá
**[Hiển thị code evaluation]**

"Hàm `_evaluate_position()` hoạt động như sau:

1. Đặt quân thử nghiệm
2. Quét 4 hướng: ngang, dọc, 2 chéo
3. Đếm số quân liên tiếp + kiểm tra đầu mở
4. Tính điểm theo bảng pattern
5. Khôi phục bàn cờ

**[Run cell 12]**

Điểm mạnh: cùng 1 hàm cho cả tấn công VÀ phòng thủ!"

---

## Slide 13-14: Tối Ưu Tỷ Lệ Tấn Công/Phòng Thủ
**[Hiển thị graph]**

"Tại sao 4.5:1.5?

**[Run cell 14]**

- 2:1 → quá thủ → 52% winrate
- 6:1 → quá công → 61% winrate  
- **4.5:1.5 → cân bằng hoàn hảo → 67% winrate!**

Đây là vùng 'Goldilocks' - vừa đủ công, vừa đủ thủ."

---

## Slide 15-16: Implementation Đầy Đủ
**[Hiển thị complete agent code]**

"Đây là agent hoàn chỉnh với cả 2 methods tích hợp.

**[Run cell 16]**

Agent đã sẵn sàng chiến đấu!"

---

## Slide 17-18: Demo Trực Tiếp
**[Run cell 18 - demo 20 games]**

"Giờ là thời điểm quyết định. 20 trận đấu thử nghiệm...

[Đợi kết quả...]

Tuyệt vời! Đạt [X]% winrate, vượt mục tiêu 60%, với thời gian trung bình chỉ [X] giây/trận!"

---

## Slide 19-20: Tối Ưu Parallel
**[Hiển thị markdown parallel]**

"Để kiểm tra thống kê, cần 100+ trận. Chúng tôi dùng multiprocessing:

- Tự động phát hiện CPU cores
- Phân phối game qua workers
- Tăng tốc 4-8x

Kết quả: giảm từ 5 phút xuống 45 giây - rất quan trọng cho thử nghiệm nhanh!"

---

## Slide 21: Tổng Kết
**[Hiển thị summary]**

"**Kiến trúc:**
- ✅ Pattern heuristic cổ điển - không cần NN
- ✅ Chiến lược 3 tầng: Thắng → Chặn → Đánh giá
- ✅ Phân tích 4 hướng

**Hiệu suất:**
- 🏆 Thắng: 65-70%
- ⚡ Tốc độ: 20 trận trong 5-10 giây
- 🚀 Parallel: 100 trận trong 10-20 giây

**Kết luận quan trọng:** Đôi khi giải pháp cổ điển được thiết kế tốt có thể sánh ngang machine learning hiện đại - mà không cần training data hay computational power khổng lồ!

## Ghi Chú Kỹ Thuật Cho Người Trình Bày

- **Thời gian:** 5 phút (đã tối ưu từ 15-20 phút)
- **Giảm thiểu rủi ro demo:** Chuẩn bị kết quả pre-recorded nếu live demo fail
- **Tùy chỉnh theo đối tượng:** 
  - Kỹ thuật: Nhấn mạnh chi tiết thuật toán
  - Kinh doanh: Tập trung kết quả, tốc độ phát triển  
  - Học thuật: So sánh với ML, hướng nghiên cứu
- **Tips tương tác:**
  - Hỏi "Ai biết cờ caro?" ngay đầu
  - Mời dự đoán winrate trước khi show kết quả
  - Thảo luận tradeoff giữa giải pháp đơn giản/phức tạp

---

## Bảng Thời Gian Chi Tiết (5 Phút)

| Thời gian | Nội dung | Slide |
|-----------|----------|-------|
| 0:00-0:30 | Giới thiệu + mục tiêu | 1 |
| 0:30-1:00 | Setup nhanh | 2-4 |
| 1:00-2:00 | Chiến lược 3 tầng + Pattern scoring | 5-10 |
| 2:00-2:45 | Thuật toán + Tối ưu tỷ lệ | 11-14 |
| 2:45-3:15 | Implementation + Demo | 15-18 |
| 3:15-4:00 | Parallel optimization | 19-20 |
| 4:00-4:45 | Tổng kết | 21 |
| 4:45-5:00 | Q&A buffer | - |
