import numpy as np
from typing import Tuple, List, Optional
import random
import time


class GomokuGame:
    def __init__(self, board_size: int = 15):
        """
        Init
        Args:
            board_size: Kích thước bàn cờ (default 15x15)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1: Đen (X), 2: Trắng (O)
        self.move_history = []

    def reset(self):
        """Reset game về trạng thái ban đầu"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.move_history = []

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Kiểm tra nước đi có hợp lệ không
        Args:
            row, col: Tọa độ nước đi
        """
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row][col] == 0

    def make_move(self, row: int, col: int) -> bool:
        """
        Thực hiện nước đi
        Args:
            row, col: Tọa độ nước đi
        Returns:
            True nếu thành công
        """
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        self.current_player = 3 - self.current_player  # Đổi người chơi (1->2, 2->1)
        return True

    def undo_move(self):
        """Hoàn tác nước đi cuối cùng"""
        if not self.move_history:
            return False

        row, col, player = self.move_history.pop()
        self.board[row][col] = 0
        self.current_player = player
        return True

    def check_winner(self, row: int, col: int) -> bool:
        """
        Kiểm tra thắng với LUẬT MỚI:
        - 5 quân liên tiếp bị chặn 2 đầu (OXXXXXO) -> KHÔNG THẮNG
        - 6+ quân liên tiếp -> THẮNG (không quan tâm bị chặn)
        - 5 quân có ít nhất 1 đầu hở -> THẮNG
        """
        player = self.board[row][col]
        if player == 0:
            return False

        opponent = 3 - player

        # 4 hướng: ngang, dọc, chéo chính, chéo phụ
        directions = [
            (0, 1),  # Ngang
            (1, 0),  # Dọc
            (1, 1),  # Chéo chính
            (1, -1)  # Chéo phụ
        ]

        for dx, dy in directions:
            count = 1  # Đếm quân tại vị trí hiện tại

            # Đếm về hướng dương (dx, dy)
            x, y = row + dx, col + dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x][y] == player:
                    count += 1
                    x += dx
                    y += dy
                else:
                    break

            # Lưu vị trí cuối cùng của hướng dương (để kiểm tra chặn)
            end_x, end_y = x, y

            # Đếm về hướng âm (-dx, -dy)
            x, y = row - dx, col - dy
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x][y] == player:
                    count += 1
                    x -= dx
                    y -= dy
                else:
                    break

            # Lưu vị trí cuối cùng của hướng âm (để kiểm tra chặn)
            start_x, start_y = x, y

            # Kiểm tra điều kiện thắng
            if count >= 6:
                # 6+ quân -> THẮNG luôn
                return True

            elif count == 5:
                # 5 quân -> Kiểm tra có bị chặn 2 đầu không

                # Kiểm tra đầu bên trái/trên (hướng âm)
                start_blocked = False
                if 0 <= start_x < self.board_size and 0 <= start_y < self.board_size:
                    if self.board[start_x][start_y] == opponent:
                        start_blocked = True
                else:
                    # Ra ngoài biên cũng tính là bị chặn
                    start_blocked = True

                # Kiểm tra đầu bên phải/dưới (hướng dương)
                end_blocked = False
                if 0 <= end_x < self.board_size and 0 <= end_y < self.board_size:
                    if self.board[end_x][end_y] == opponent:
                        end_blocked = True
                else:
                    # Ra ngoài biên cũng tính là bị chặn
                    end_blocked = True

                # Nếu CẢ 2 đầu đều bị chặn -> KHÔNG THẮNG
                if start_blocked and end_blocked:
                    continue  # Thử hướng khác
                else:
                    # Có ít nhất 1 đầu hở -> THẮNG
                    return True

        return False

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        Kiểm tra trò chơi kết thúc chưa
        Returns:
            (game_over, winner):
                - game_over: True nếu kết thúc
                - winner: 1 hoặc 2 nếu có người thắng, 0 nếu hòa, None nếu chưa kết thúc
        """
        # Kiểm tra nước đi cuối có thắng không
        if self.move_history:
            last_row, last_col, last_player = self.move_history[-1]
            if self.check_winner(last_row, last_col):
                return True, last_player

        # Kiểm tra hòa (bàn cờ đầy)
        if len(self.move_history) == self.board_size * self.board_size:
            return True, 0

        return False, None

    def display(self):
        """Hiển thị bàn cờ"""
        print("\n  ", end="")
        for i in range(self.board_size):
            print(f"{i:2}", end=" ")
        print()

        for i in range(self.board_size):
            print(f"{i:2} ", end="")
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    print(" . ", end="")
                elif self.board[i][j] == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
            print()
        print()

    def get_board_copy(self):
        """Lấy bản sao của bàn cờ"""
        return self.board.copy()


class SmartRandomAgent:
    def __init__(self, name="SmartRandom"):
        self.name = name

    def get_move(self, game: GomokuGame) -> Tuple[int, int]:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # Nước đi đầu tiên - đánh giữa
        if len(game.move_history) == 0:
            center = game.board_size // 2
            return (center, center)

        my_player = game.current_player
        opponent = 3 - my_player

        # 1. Kiểm tra xem có nước nào THẮNG NGAY không
        for move in valid_moves:
            row, col = move
            game.board[row][col] = my_player
            if game.check_winner(row, col):
                game.board[row][col] = 0
                return move  # THẮNG NGAY!
            game.board[row][col] = 0

        # 2. Kiểm tra xem đối thủ có nước nào thắng ngay không -> CHẶN
        for move in valid_moves:
            row, col = move
            game.board[row][col] = opponent
            if game.check_winner(row, col):
                game.board[row][col] = 0
                return move  # CHẶN ĐỐI THỦ
            game.board[row][col] = 0

        # 3. Đánh giá các nước đi theo pattern
        scored_moves = []

        for move in valid_moves:
            row, col = move
            score = 0

            # Đánh giá tấn công (mình đánh vào đây tốt như thế nào)
            attack_score = self._evaluate_position(game, row, col, my_player)

            # Đánh giá phòng thủ (chặn đối thủ)
            defense_score = self._evaluate_position(game, row, col, opponent)

            # Tổng hợp (ưu tiên phòng thủ hơn)
            score = attack_score + defense_score * 1.3

            # Cộng điểm cho vị trí gần trung tâm
            center = game.board_size // 2
            distance_to_center = abs(row - center) + abs(col - center)
            score -= distance_to_center * 5

            scored_moves.append((move, score))

        # Chọn nước đi tốt nhất (có thêm yếu tố ngẫu nhiên)
        if scored_moves:
            best_moves = sorted(scored_moves, key=lambda x: x[1], reverse=True)
            top_moves = best_moves[:min(5, len(best_moves))]

            scores = [move[1] for move in top_moves]
            min_val = min(scores)

            if min_val <= 0:
                offset = abs(min_val) + 1
                weights = [s + offset for s in scores]
            else:
                weights = scores

            return random.choices([m[0] for m in top_moves], weights=weights, k=1)[0]

        return random.choice(valid_moves)

    def _evaluate_position(self, game, row: int, col: int, player: int) -> float:
        """
        Đánh giá một vị trí cho một người chơi cụ thể
        """
        game.board[row][col] = player

        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            open_ends = 0

            # Đếm về phía trước
            x, y = row + dx, col + dy
            while 0 <= x < game.board_size and 0 <= y < game.board_size:
                if game.board[x][y] == player:
                    count += 1
                    x += dx
                    y += dy
                elif game.board[x][y] == 0:
                    open_ends += 1
                    break
                else:
                    break

            # Đếm về phía sau
            x, y = row - dx, col - dy
            while 0 <= x < game.board_size and 0 <= y < game.board_size:
                if game.board[x][y] == player:
                    count += 1
                    x -= dx
                    y -= dy
                elif game.board[x][y] == 0:
                    open_ends += 1
                    break
                else:
                    break

            # Tính điểm dựa trên pattern
            if count >= 5:
                score += 100000
            elif count == 4:
                if open_ends == 2:
                    score += 10000
                elif open_ends == 1:
                    score += 1000
            elif count == 3:
                if open_ends == 2:
                    score += 500
                elif open_ends == 1:
                    score += 100
            elif count == 2:
                if open_ends == 2:
                    score += 50
                elif open_ends == 1:
                    score += 10

        game.board[row][col] = 0
        return score


class MinimaxAgent:
    def __init__(self, name="Minimax_SuperDef", max_depth=3, search_radius=2):
        self.name = name
        self.max_depth = max_depth
        self.search_radius = search_radius
        self.nodes_explored = 0

    def get_move(self, game: GomokuGame) -> Tuple[int, int]:
        """
        Tìm nước đi tối ưu:
        1. Safety Layer: Kiểm tra thắng ngay/thua ngay.
        2. Minimax: Duyệt cây tìm kiếm.
        """
        self.nodes_explored = 0
        valid_moves = self._get_smart_moves(game)
        if not valid_moves:
            return None

        # Nước đầu tiên luôn đánh giữa
        if len(game.move_history) == 0:
            c = game.board_size // 2
            return (c, c)

        my_player = game.current_player
        opponent = 3 - my_player

        # Check 1: KILL MOVE (Có nước thắng -> Đánh ngay)
        for r, c in valid_moves:
            game.board[r][c] = my_player
            if game.check_winner(r, c):
                game.board[r][c] = 0
                return (r, c)
            game.board[r][c] = 0

        # Check 2: BLOCK WIN (Đối thủ sắp thắng -> Chặn ngay)
        forced_blocks = []
        for r, c in valid_moves:
            game.board[r][c] = opponent
            if game.check_winner(r, c):
                forced_blocks.append((r, c))
            game.board[r][c] = 0

        if forced_blocks:
            # Nếu bắt buộc phải chặn, chọn nước chặn tốt nhất (theo hàm đánh giá nhanh)
            # Hoặc đơn giản là lấy nước đầu tiên tìm thấy
            return forced_blocks[0]

        # =================================================================
        # 2. MINIMAX ALGORITHM
        # =================================================================

        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Sắp xếp nước đi để cắt tỉa Alpha-Beta hiệu quả hơn
        sorted_moves = self._sort_moves_by_priority(game, valid_moves, my_player)

        # BEAM SEARCH: Chỉ lấy Top 20 nước đi tốt nhất để duyệt sâu
        # Giúp code chạy nhanh hơn và tập trung vào các nước quan trọng
        moves_to_search = sorted_moves[:20] if len(sorted_moves) > 20 else sorted_moves

        for move in moves_to_search:
            r, c = move
            game.make_move(r, c)

            # Gọi đệ quy Minimax
            value = self._minimax(game, self.max_depth - 1, alpha, beta, False, my_player)

            game.undo_move()

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_move if best_move else (valid_moves[0] if valid_moves else None)

    def _minimax(self, game, depth, alpha, beta, is_maximizing, original_player):
        self.nodes_explored += 1

        # Kiểm tra kết thúc game
        game_over, winner = game.is_game_over()
        if game_over:
            if winner == original_player:
                return 100000000 + depth * 1000  # Thắng càng sớm càng tốt
            elif winner == 0:
                return 0
            else:
                return -100000000 - depth * 1000  # Thua càng muộn càng tốt

        if depth == 0:
            return self.evaluate_board(game, original_player)

        # Lấy nước đi
        moves = self._get_smart_moves(game)
        if not moves: return 0

        # Sắp xếp sơ bộ (chỉ cần thiết ở các tầng trên cao, tầng lá không cần thiết lắm)
        if depth >= 2:
            current_turn = original_player if is_maximizing else (3 - original_player)
            moves = self._sort_moves_by_priority(game, moves, current_turn)[:15]  # Cắt tỉa mạnh

        if is_maximizing:
            best_value = float('-inf')
            for r, c in moves:
                game.make_move(r, c)
                value = self._minimax(game, depth - 1, alpha, beta, False, original_player)
                game.undo_move()
                best_value = max(best_value, value)
                alpha = max(alpha, value)
                if beta <= alpha: break
            return best_value
        else:
            best_value = float('inf')
            for r, c in moves:
                game.make_move(r, c)
                value = self._minimax(game, depth - 1, alpha, beta, True, original_player)
                game.undo_move()
                best_value = min(best_value, value)
                beta = min(beta, value)
                if beta <= alpha: break
            return best_value

    def evaluate_board(self, game: GomokuGame, original_player: int) -> float:
        """
        Hàm đánh giá tối ưu cho Depth 3:
        Ưu tiên PHÒNG THỦ GẤP 4 LẦN.
        """
        opponent = 3 - original_player

        # Phân tích bàn cờ
        my_stats = self._analyze_threats(game, original_player)
        opp_stats = self._analyze_threats(game, opponent)

        # 1. ĐIỀU KIỆN THẮNG/THUA CHẮC CHẮN (Checkmate)
        if my_stats['five'] > 0: return 100000000
        if opp_stats['five'] > 0: return -100000000

        if my_stats['open_four'] > 0: return 90000000
        if opp_stats['open_four'] > 0: return -95000000  # Đối thủ có Open 4 -> Coi như thua

        # 2. TÍNH ĐIỂM
        # Điểm tấn công
        my_score = (my_stats['four'] * 10000 +
                    my_stats['open_three'] * 8000 +  # Open 3 rất giá trị
                    my_stats['three'] * 500 +
                    my_stats['open_two'] * 100)

        # Điểm đe dọa của đối thủ
        opp_score = (opp_stats['four'] * 10000 +
                     opp_stats['open_three'] * 8000 +
                     opp_stats['three'] * 500 +
                     opp_stats['open_two'] * 100)

        # TRỌNG SỐ PHÒNG THỦ:
        # Depth thấp -> "Mù" tương lai -> Phải sợ đối thủ hơn bình thường.
        # Hệ số 4.0 đảm bảo AI sẽ bỏ tấn công để quay về thủ nếu đối thủ có Open 3.
        return my_score - (opp_score * 4.0)

    def _analyze_threats(self, game, player) -> dict:
        """
        Đếm số lượng pattern trên bàn cờ.
        Đảm bảo KHÔNG ĐẾM TRÙNG (mỗi chuỗi chỉ tính 1 lần).
        """
        stats = {
            'five': 0, 'open_four': 0, 'four': 0,
            'open_three': 0, 'three': 0, 'open_two': 0
        }
        size = game.board_size
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        visited = set()  # (r, c, direction_index)

        for r in range(size):
            for c in range(size):
                if game.board[r][c] != player:
                    continue

                for dir_idx, (dx, dy) in enumerate(directions):
                    if (r, c, dir_idx) in visited:
                        continue

                    # Bắt đầu đếm chuỗi
                    count = 0
                    curr_r, curr_c = r, c
                    while 0 <= curr_r < size and 0 <= curr_c < size and game.board[curr_r][curr_c] == player:
                        count += 1
                        visited.add((curr_r, curr_c, dir_idx))
                        curr_r += dx
                        curr_c += dy

                    # Kiểm tra 2 đầu chặn
                    # Đầu cuối (curr_r, curr_c)
                    end_blocked = True
                    if 0 <= curr_r < size and 0 <= curr_c < size and game.board[curr_r][curr_c] == 0:
                        end_blocked = False

                    # Đầu đầu (r-dx, c-dy)
                    start_r, start_c = r - dx, c - dy
                    start_blocked = True
                    if 0 <= start_r < size and 0 <= start_c < size and game.board[start_r][start_c] == 0:
                        start_blocked = False

                    open_ends = (1 if not start_blocked else 0) + (1 if not end_blocked else 0)

                    # Phân loại
                    if count >= 5:
                        stats['five'] += 1
                    elif count == 4:
                        if open_ends == 2:
                            stats['open_four'] += 1
                        elif open_ends == 1:
                            stats['four'] += 1
                    elif count == 3:
                        if open_ends == 2:
                            stats['open_three'] += 1
                        elif open_ends == 1:
                            stats['three'] += 1
                    elif count == 2:
                        if open_ends == 2: stats['open_two'] += 1

        return stats

    def _get_smart_moves(self, game: GomokuGame) -> List[Tuple[int, int]]:
        """Lấy các nước đi xung quanh các quân cờ hiện có"""
        if len(game.move_history) == 0:
            c = game.board_size // 2
            return [(c, c)]

        occupied = set()
        for r in range(game.board_size):
            for c in range(game.board_size):
                if game.board[r][c] != 0:
                    occupied.add((r, c))

        candidates = set()
        for r, c in occupied:
            for dr in range(-self.search_radius, self.search_radius + 1):
                for dc in range(-self.search_radius, self.search_radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < game.board_size and 0 <= cc < game.board_size:
                        if game.board[rr][cc] == 0:
                            candidates.add((rr, cc))

        if not candidates: return game.get_valid_moves()
        return list(candidates)

    def _sort_moves_by_priority(self, game, moves, player):
        """Sắp xếp nước đi để Minimax cắt tỉa tốt hơn"""
        scored = []
        opponent = 3 - player
        center = game.board_size // 2

        for r, c in moves:
            p = 0
            # Ưu tiên gần tâm
            p -= (abs(r - center) + abs(c - center))

            # Đánh giá nhanh tấn công
            p += self._quick_evaluate(game, r, c, player)
            # Đánh giá nhanh phòng thủ (quan trọng)
            p += self._quick_evaluate(game, r, c, opponent) * 1.5

            scored.append((p, (r, c)))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored]

    def _quick_evaluate(self, game, row, col, player):
        """Hàm đánh giá cục bộ cực nhanh dùng cho việc sort"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            open_ends = 0

            # Check 2 hướng
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

            if count >= 4:
                score += 10000
            elif count == 3:
                score += 1000 if open_ends > 0 else 0
            elif count == 2 and open_ends == 2:
                score += 100

        return score

def play_game(agent1, agent2, display=True, game_number=None) -> Tuple[int, int]:
    """
    Cho 2 agent chơi với nhau
    Args:
        display: Nếu True sẽ in bàn cờ ra màn hình mỗi lượt
    """
    game = GomokuGame()
    agents = [agent1, agent2]

    if display:
        header = f"Game {game_number}: {agent1.name} (X) vs {agent2.name} (O)" if game_number else f"Game: {agent1.name} (X) vs {agent2.name} (O)"
        print(f"\n{'=' * 70}")
        print(header)
        print(f"{'=' * 70}")
        print("Bàn cờ ban đầu:")
        game.display()

    move_count = 0
    start_time = time.time()

    while True:
        current_agent = agents[game.current_player - 1]
        player_symbol = "X" if game.current_player == 1 else "O"

        # Agent tìm nước đi
        move = current_agent.get_move(game)
        if move is None:
            print("Hết nước đi (Hòa)!")
            break

        row, col = move

        # Thực hiện nước đi
        if not game.make_move(row, col):
            if display:
                print(f"Nước đi lỗi của {current_agent.name}: ({row}, {col})")
            break

        move_count += 1

        if display:
            print(f"\n{'─' * 40}")
            print(f"Nước {move_count}: {current_agent.name} ({player_symbol}) đánh vào ({row}, {col})")
            game.display()

        game_over, winner = game.is_game_over()
        if game_over:
            elapsed_time = time.time() - start_time
            if display:
                print(f"{'═' * 70}")
                if winner == 0:
                    print(f"HÒA sau {move_count} nước! (Time: {elapsed_time:.2f}s)")
                else:
                    winner_name = agents[winner - 1].name
                    print(f"{winner_name} THẮNG sau {move_count} nước! (Time: {elapsed_time:.2f}s)")
                print(f"{'═' * 70}\n")
            return winner, move_count

    return 0, move_count


def test_agents_detailed(agent1, agent2, num_games=2):
    wins = {1: 0, 2: 0, 0: 0}

    print(f"\n{'=' * 80}")
    print(f"BẮT ĐẦU TRẬN ĐẤU: {agent1.name} vs {agent2.name}")
    print(f"{'=' * 80}\n")

    for i in range(num_games):
        game_num = i + 1

        if i % 2 == 0:
            winner, moves = play_game(agent1, agent2, display=True, game_number=game_num)
            if winner == 1:
                wins[1] += 1
            elif winner == 2:
                wins[2] += 1
            else:
                wins[0] += 1
        else:
            winner, moves = play_game(agent2, agent1, display=True, game_number=game_num)
            if winner == 1:
                wins[2] += 1
            elif winner == 2:
                wins[1] += 1
            else:
                wins[0] += 1

    print(f"\n{'=' * 80}")
    print(f"KẾT QUẢ TỔNG HỢP ({num_games} games)")
    print(f"{'=' * 80}")
    print(f"{agent1.name} thắng: {wins[1]}")
    print(f"{agent2.name} thắng: {wins[2]}")
    print(f"Hòa: {wins[0]}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    NUM_GAMES = 10
    TEST_DEPTH = 3


    minimax_ai = MinimaxAgent(name="Minimax_Agent", max_depth=TEST_DEPTH, search_radius=2)

    opponent = SmartRandomAgent(name="Random_Agent")

    test_agents_detailed(minimax_ai, opponent, num_games=NUM_GAMES)