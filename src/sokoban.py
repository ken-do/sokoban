import heapq
from collections import deque
import time
import sys

class SokobanState:
    def __init__(self, player_pos, boxes, walls, goals, parent=None, move=None):
        self.player_pos = player_pos
        self.boxes = frozenset(boxes)
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.parent = parent
        self.move = move

    def is_goal(self):
        return self.boxes == self.goals

    def __hash__(self):
        return hash((self.player_pos, self.boxes))

    def __eq__(self, other):
        return self.player_pos == other.player_pos and self.boxes == other.boxes

    def __lt__(self, other):
        return False


DIRECTIONS = {
    'U': (0, -1),  # Up
    'D': (0, 1),  # Down
    'L': (-1, 0),  # Left
    'R': (1, 0)  # Right
}


def parse_test_case(test_case):
    """
    # : wall
    @ : player
    $ : box
    . : goal
    * : box on goal
    + : player on goal
    """
    player_pos = None
    boxes = set()
    walls = set()
    goals = set()

    for y, row in enumerate(test_case):
        for x, char in enumerate(row):
            if char == '#':
                walls.add((x, y))
            elif char == '@':
                player_pos = (x, y)
            elif char == '$':
                boxes.add((x, y))
            elif char == '.':
                goals.add((x, y))
            elif char == '*':
                boxes.add((x, y))
                goals.add((x, y))
            elif char == '+':
                player_pos = (x, y)
                goals.add((x, y))

    return player_pos, boxes, walls, goals


def is_deadlock(box_pos, boxes, walls, goals):
    """
    Kiểm tra deadlock: hộp bị kẹt không thể di chuyển đến đích
    - Corner deadlock: hộp ở góc tường
    - Edge deadlock: hộp dính tường và không có goal trên cạnh đó
    """
    x, y = box_pos

    # Nếu hộp đã ở đích thì không deadlock
    if box_pos in goals:
        return False

    # Kiểm tra corner deadlock: hộp ở góc (2 cạnh kề nhau bị chặn bởi tường)
    if ((x - 1, y) in walls and (x, y - 1) in walls) or \
            ((x + 1, y) in walls and (x, y - 1) in walls) or \
            ((x - 1, y) in walls and (x, y + 1) in walls) or \
            ((x + 1, y) in walls and (x, y + 1) in walls):
        return True

    # Kiểm tra edge deadlock: hộp dính tường ngang và không có goal nào trên hàng đó
    if (x - 1, y) in walls or (x + 1, y) in walls:
        # Kiểm tra xem có goal nào trên cùng hàng không
        has_goal_on_row = any(g[1] == y for g in goals)
        if not has_goal_on_row:
            # Kiểm tra xem hộp có thể di chuyển lên/xuống không
            if ((x, y - 1) in walls or (x, y - 1) in boxes) and \
                    ((x, y + 1) in walls or (x, y + 1) in boxes):
                return True

    # Kiểm tra edge deadlock: hộp dính tường dọc và không có goal nào trên cột đó
    if (x, y - 1) in walls or (x, y + 1) in walls:
        # Kiểm tra xem có goal nào trên cùng cột không
        has_goal_on_col = any(g[0] == x for g in goals)
        if not has_goal_on_col:
            # Kiểm tra xem hộp có thể di chuyển trái/phải không
            if ((x - 1, y) in walls or (x - 1, y) in boxes) and \
                    ((x + 1, y) in walls or (x + 1, y) in boxes):
                return True

    return False


def get_successors(state):
    """Lấy tất cả các trạng thái kế tiếp hợp lệ"""
    successors = []
    x, y = state.player_pos

    for direction, (dx, dy) in DIRECTIONS.items():
        new_x, new_y = x + dx, y + dy
        new_pos = (new_x, new_y)

        # Kiểm tra va chạm tường
        if new_pos in state.walls:
            continue

        # Kiểm tra có hộp không
        if new_pos in state.boxes:
            # Vị trí mới của hộp nếu đẩy
            box_new_x, box_new_y = new_x + dx, new_y + dy
            box_new_pos = (box_new_x, box_new_y)

            # Kiểm tra xem có thể đẩy hộp không
            if box_new_pos in state.walls or box_new_pos in state.boxes:
                continue

            # Tạo trạng thái mới với hộp được đẩy
            new_boxes = set(state.boxes)
            new_boxes.remove(new_pos)
            new_boxes.add(box_new_pos)

            # Kiểm tra deadlock
            if is_deadlock(box_new_pos, new_boxes, state.walls, state.goals):
                continue

            new_state = SokobanState(new_pos, new_boxes, state.walls,
                                     state.goals, state, direction)
            successors.append(new_state)
        else:
            # Di chuyển bình thường không đẩy hộp
            new_state = SokobanState(new_pos, state.boxes, state.walls,
                                     state.goals, state, direction)
            successors.append(new_state)

    return successors


def bfs_search(initial_state):
    """
    Thuật toán BFS - Tìm kiếm theo chiều rộng
    - Sử dụng queue (FIFO) để duyệt các trạng thái
    - Đảm bảo tìm được lời giải ngắn nhất (nếu có)
    """
    print("\n=== BẮT ĐẦU TÌM KIẾM VỚI BFS ===")

    start_time = time.time()
    queue = deque([initial_state])
    visited = {initial_state}
    nodes_expanded = 0
    max_queue_size = 1

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        current_state = queue.popleft()
        nodes_expanded += 1

        # In tiến trình mỗi 1000 nodes
        if nodes_expanded % 1000 == 0:
            print(f"Đã duyệt {nodes_expanded} nodes, queue size: {len(queue)}")

        # Kiểm tra đích
        if current_state.is_goal():
            elapsed_time = time.time() - start_time
            print(f"\n✓ Tìm thấy lời giải!")
            print(f"Thời gian: {elapsed_time:.2f}s")
            print(f"Số nodes đã duyệt: {nodes_expanded}")
            print(f"Kích thước queue tối đa: {max_queue_size}")
            visited_count = len(visited)
            return reconstruct_path(current_state), nodes_expanded, elapsed_time, max_queue_size, visited_count

        # Tạo các trạng thái kế tiếp
        for successor in get_successors(current_state):
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    elapsed_time = time.time() - start_time
    print(f"\n✗ Không tìm thấy lời giải sau {elapsed_time:.2f}s")
    visited_count = len(visited)
    return None, nodes_expanded, elapsed_time, max_queue_size, visited_count


def manhattan_distance(pos1, pos2):
    """Tính khoảng cách Manhattan giữa 2 điểm"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def heuristic(state):
    """
    Hàm heuristic cho A*
    - Tính tổng khoảng cách Manhattan từ mỗi hộp đến đích gần nhất
    - Admissible: không bao giờ overestimate chi phí thực tế
    """
    if not state.boxes:
        return 0

    total = 0
    boxes_list = list(state.boxes)
    goals_list = list(state.goals)

    # Với mỗi hộp, tìm đích gần nhất
    for box in boxes_list:
        min_dist = float('inf')
        for goal in goals_list:
            dist = manhattan_distance(box, goal)
            min_dist = min(min_dist, dist)
        total += min_dist

    return total


def astar_search(initial_state):
    """
    Thuật toán A* - Tìm kiếm có thông tin
    - Sử dụng priority queue với f(n) = g(n) + h(n)
    - g(n): chi phí từ start đến n
    - h(n): ước lượng chi phí từ n đến goal
    """
    print("\n=== BẮT ĐẦU TÌM KIẾM VỚI A* ===")

    start_time = time.time()

    # Priority queue: (f_score, counter, state)
    counter = 0
    open_set = [(heuristic(initial_state), counter, initial_state)]
    counter += 1

    # g_score: chi phí từ start đến mỗi node
    g_score = {initial_state: 0}
    visited = set()
    nodes_expanded = 0
    max_heap_size = 1

    while open_set:
        max_heap_size = max(max_heap_size, len(open_set))
        _, _, current_state = heapq.heappop(open_set)

        if current_state in visited:
            continue

        visited.add(current_state)
        nodes_expanded += 1

        # In tiến trình mỗi 500 nodes
        if nodes_expanded % 500 == 0:
            print(f"Đã duyệt {nodes_expanded} nodes, heap size: {len(open_set)}")

        # Kiểm tra đích
        if current_state.is_goal():
            elapsed_time = time.time() - start_time
            print(f"\n✓ Tìm thấy lời giải!")
            print(f"Thời gian: {elapsed_time:.2f}s")
            print(f"Số nodes đã duyệt: {nodes_expanded}")
            print(f"Kích thước heap tối đa: {max_heap_size}")
            g_score_size = len(g_score)
            return reconstruct_path(current_state), nodes_expanded, elapsed_time, max_heap_size, g_score_size

        current_g = g_score[current_state]

        # Xét các trạng thái kế tiếp
        for successor in get_successors(current_state):
            tentative_g = current_g + 1

            if successor not in g_score or tentative_g < g_score[successor]:
                g_score[successor] = tentative_g
                f_score = tentative_g + heuristic(successor)
                heapq.heappush(open_set, (f_score, counter, successor))
                counter += 1

    elapsed_time = time.time() - start_time
    print(f"\n✗ Không tìm thấy lời giải sau {elapsed_time:.2f}s")
    g_score_size = len(g_score)
    return None, nodes_expanded, elapsed_time, max_heap_size, g_score_size

def reconstruct_path(state):
    """Tái tạo đường đi từ trạng thái đích về trạng thái đầu"""
    path = []
    current = state
    while current.parent is not None:
        path.append(current.move)
        current = current.parent
    path.reverse()
    return path



def print_comparison_tables(bfs_result, astar_result):
    """In ra hai bảng so sánh trên console:
    - Bảng so sánh thời gian
    - Bảng so sánh bộ nhớ

    Mỗi dict kết quả được kỳ vọng sẽ chứa các khóa: 'time', 'nodes', 'length', 'frontier'.
    """
    # Chuẩn bị giá trị với định dạng an toàn và khớp chính xác với tiêu đề markdown trong các file ví dụ
    bfs_time = bfs_result.get('time', 0.0)
    astar_time = astar_result.get('time', 0.0)
    bfs_nodes = bfs_result.get('nodes', 0)
    astar_nodes = astar_result.get('nodes', 0)
    bfs_len = bfs_result.get('length', 0)
    astar_len = astar_result.get('length', 0)
    bfs_frontier = bfs_result.get('frontier', 0)
    astar_frontier = astar_result.get('frontier', 0)
    bfs_visited = bfs_result.get('visited', bfs_nodes)
    astar_gscore = astar_result.get('gscore_size', 0)

    # Kích thước bảng và tên test case được kỳ vọng có trong dict kết quả (hàm main sẽ thêm vào)
    test_case_name = bfs_result.get('test_name', 'Test Case')
    table_size = bfs_result.get('table_size', '')
    num_boxes = bfs_result.get('num_boxes', '')

    # Tính toán tỉ lệ tăng tốc (speedup) và phần trăm giảm số node một cách an toàn
    speedup = (bfs_time / astar_time) if astar_time and astar_time != 0 else float('inf')
    node_reduction = (1 - (astar_nodes / bfs_nodes)) * 100 if bfs_nodes and bfs_nodes != 0 else 0.0

    # Định dạng và in ra bảng SO SÁNH THỜI GIAN
    print('\n' + '=' * 60)
    print('TIME COMPARISON')
    print('=' * 60)
    print('| Test Case        | Table size | No. of boxes | BFS Time (s) | BFS Nodes | A* Time (s) | A* Nodes | Speedup | Node Reduction |')
    print('|------------------|-------------|---------------|---------------|------------|--------------|-----------|----------|----------------|')
    print(f"| {test_case_name:<16} | {table_size:<11} | {str(num_boxes):<13} | {bfs_time:0.3f}         | {bfs_nodes:<10} | {astar_time:0.3f}        | {astar_nodes:<9} | {speedup:0.1f}x     | {node_reduction:0.1f}%          |")

    # Bảng so sánh bộ nhớ
    # Bộ nhớ (MB) ≈ (Số node đã duyệt × 280 + Frontier × 280 + G-Score × 8) / (1024*1024)
    NODE_SIZE = 280  # số byte cho mỗi trạng thái trong visited/frontier
    GSCORE_SIZE = 8  # số byte cho mỗi phần tử trong g-score
    
    # Bộ nhớ BFS: node đã duyệt + hàng đợi frontier (không có g-score)
    bfs_memory_bytes = (bfs_visited * NODE_SIZE) + (bfs_frontier * NODE_SIZE)
    bfs_memory_mb = bfs_memory_bytes / (1024 * 1024)
    
    # Bộ nhớ A*: node đã duyệt + heap frontier + dictionary g-score
    astar_memory_bytes = (astar_nodes * NODE_SIZE) + (astar_frontier * NODE_SIZE) + (astar_gscore * GSCORE_SIZE)
    astar_memory_mb = astar_memory_bytes / (1024 * 1024)
    
    memory_reduction = (1 - (astar_memory_mb / bfs_memory_mb)) * 100 if bfs_memory_mb and bfs_memory_mb != 0 else 0.0

    # Định dạng và in ra bảng SO SÁNH TIÊU TỐN BỘ NHỚ
    print('\n' + '=' * 60)
    print('MEMORY COMPARISON')
    print('=' * 60)
    print('| Test Case     | BFS Visited States | BFS Max Queue Size | BFS Memory Usage (MB) | A* Visited States | A* Max Heap Size | A* G-Score Dict Size | A* Memory Usage (MB) | Memory Reduction |')
    print('|----------------|--------------------|--------------------|------------------------|-------------------|------------------|----------------------|----------------------|------------------|')
    # Sử dụng định dạng hợp lệ: căn lề, độ rộng và dấu phân cách hàng nghìn
    print(f"| {test_case_name:<14} | {bfs_visited:>18,} | {bfs_frontier:>18,} | {bfs_memory_mb:>22.1f} | {astar_nodes:>15,} | {astar_frontier:>16,} | {astar_gscore:>20,} | {astar_memory_mb:>22.1f} | {memory_reduction:>16.1f}% |")


def print_state(state, width, height):
    """In trạng thái game ra console"""
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Vẽ tường
    for x, y in state.walls:
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = '#'

    # Vẽ đích
    for x, y in state.goals:
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = '.'

    # Vẽ hộp
    for x, y in state.boxes:
        if 0 <= y < height and 0 <= x < width:
            if (x, y) in state.goals:
                grid[y][x] = '*'  # Hộp trên đích
            else:
                grid[y][x] = '$'

    # Vẽ người chơi
    x, y = state.player_pos
    if 0 <= y < height and 0 <= x < width:
        if (x, y) in state.goals:
            grid[y][x] = '+'  # Người chơi trên đích
        else:
            grid[y][x] = '@'

    # In ra
    for row in grid:
        print(''.join(row))
    print()


def animate_solution(initial_state, moves):
    """Hiển thị từng bước của lời giải"""
    print("\n" + "=" * 50)
    print("DEMO TỪNG BƯỚC GIẢI")
    print("=" * 50)

    # Tính kích thước grid
    all_positions = list(initial_state.walls) + [initial_state.player_pos]
    width = max(x for x, y in all_positions) + 2
    height = max(y for x, y in all_positions) + 2

    current_state = initial_state
    print(f"\nTrạng thái ban đầu:")
    print_state(current_state, width, height)

    for i, move in enumerate(moves):
        successors = get_successors(current_state)
        for succ in successors:
            if succ.move == move:
                current_state = succ
                break

        print(f"Bước {i + 1}: {move}")
        print_state(current_state, width, height)

        if i < len(moves) - 1:
            input("Nhấn Enter để tiếp tục...")

    print("✓ Hoàn thành!")


TEST_CASES = {
    # --- MICROBAN (10) ---
    # Các bài tập kinh điển từ bộ Microban; độ khó từ dễ đến trung bình
    "Microban 1": [
        "####",
        "# .#",
        "#  ###",
        "#*@  #",
        "#  $ #",
        "#  ###",
        "####"
    ],

    "Microban 5": [
        " #######",
        " #     #",
        " # .$. #",
        "## $@$ #",
        "#  .$. #",
        "#      #",
        "########"
    ],

    "Microban 7": [
        "#######",
        "#     #",
        "# .$. #",
        "# $.$ #",
        "# .$. #",
        "# $.$ #",
        "#  @  #",
        "#######"
    ],

    "Microban 10": [
        "      #####",
        "      #.  #",
        "      #.# #",
        "#######.# #",
        "# @ $ $ $ #",
        "# # # # ###",
        "#       #",
        "#########"
    ],

    "Microban 15": [
        "     ###",
        "######@##",
        "#    .* #",
        "#   #   #",
        "#####$# #",
        "    #   #",
        "    #####"
    ],

    "Microban 25": [
        " ####",
        " #  ###",
        " # $$ #",
        "##... #",
        "#  @$ #",
        "#   ###",
        "#####"
    ],

    "Microban 36": [
        "####",
        "#  ############",
        "# $ $ $ $ $ @ #",
        "# .....       #",
        "###############"
    ],

    "Microban 56": [
        "#####",
        "#   ###",
        "#  $  #",
        "##* . #",
        " #   @#",
        " ######"
    ],

    "Microban 70": [
        "#####",
        "# @ ####",
        "#      #",
        "# $ $$ #",
        "##$##  #",
        "#   ####",
        "# ..  #",
        "##..  #",
        " ###  #",
        "   ####"
    ],

    "Microban 93": [
        " #########",
        "##   #   ##",
        "#    #    #",
        "#  $ # $  #",
        "#   *.*   #",
        "####.@.####",
        "#   *.*   #",
        "#  $ # $  #",
        "#    #    #",
        "##   #   ##",
        " #########"
    ],
    # --- MINICOSMOS (2) ---
    # Đơn giản với ít hộp và đích; dễ giải quyết
    "Minicosmos 03": [
        "  #####",
        "###   #",
        "# $ # ##",
        "# #  . #",
        "# .  # #",
        "##$#.$ #",
        " #@  ###",
        " #####"
    ],
    "Minicosmos 15": [
        "   ####",
        "####  #",
        "# $   #",
        "#  .# ##",
        "## #.  #",
        "# @  $ #",
        "#   ####",
        "#####"
    ],

    # --- MICROCOSMOS (2) ---
    # Phức tạp hơn với nhiều hộp và đích; yêu cầu chiến lược tốt hơn
    "Microcosmos 02": [
        "  ####",
        "###  #",
        "# $  #",
        "#  .###",
        "## #  #",
        " #@  ##",
        "  ####"
    ],
    "Microcosmos 07": [
        "#####",
        "#@$.#",
        "# $ #",
        "# . #",
        "#####"
    ],

    # --- NABOKOSMOS (3) ---
    # Phức tạp với nhiều hộp và đích; yêu cầu chiến lược nâng cao và tránh deadlock
    "Nabokosmos 04": [
        "#########",
        "# .@.$. #",
        "# # $   #",
        "# ##$####",
        "#   $  ##",
        "### .   ##",
        "  ###    #",
        "    ##   #",
        "     #####"
    ],
    "Nabokosmos 10": [
        " ####",
        " #  ######",
        " #       #",
        "## # #.$ #",
        "#  * ## ##",
        "# ** # @#",
        "###   * #",
        "  ####  #",
        "     ####"
    ],
    "Nabokosmos 21": [
        "  #####",
        " ##   # ",
        "##  #.##",
        "# @ $  #",
        "# * *  ###",
        "##*#*#   #",
        " #       #",
        " ##  #####",
        "  ####"
    ],

    # --- PICOKOSMOS (3) ---
    # Rất phức tạp với nhiều hộp và đích; yêu cầu chiến lược tinh vi và tối ưu hóa
    "Picokosmos 02": [
        "       #####",
        "     ###   #",
        "    ## . # #",
        "   ##  $$  #",
        "  ## *$. ###",
        " ## $  ###",
        "## .$. #",
        "#  @ ###",
        "#  . #",
        "######"
    ],
    "Picokosmos 06": [
        "  ####",
        " ##  ###",
        "##  .  #",
        "# $.$.$##",
        "# .$.$. #",
        "##$ #.$ #",
        " #  @  ##",
        " #######"
    ],
    "Picokosmos 14": [
        "  #######",
        " ##  #  #",
        " #      #",
        "## # #  #",
        "#  *** ##",
        "#@#*  $#",
        "#  *** #",
        "### .  #",
        "  ###  #",
        "    ####"
    ]
}

def main():
    print("=" * 60)
    print("SOKOBAN SOLVER - BFS & A*")
    print("=" * 60)
    print("Ý nghĩa các ký tự: # (tường), @ (player), $ (box), . (goal), * (box on goal)")

    # Chọn test case
    print("\nCác test case có sẵn:")
    test_names = list(TEST_CASES.keys())
    for i, name in enumerate(test_names, 1):
        print(f"{i}. {name}")

    print(f"{len(test_names) + 1}. Tự nhập test case (custom)")

    choice = input(f"\nChọn test case (1-{len(test_names) + 1}): ")

    try:
        choice_num = int(choice)
        if choice_num == len(test_names) + 1:
            # Nhập custom test case
            print("\nNhập trạng thái ban đầu (mỗi dòng một hàng, nhấn Enter 2 lần để kết thúc):")
            test_case = []
            while True:
                line = input()
                if not line:
                    break
                test_case.append(list(line))
        else:
            test_name = test_names[choice_num - 1]
            test_case = TEST_CASES[test_name]
    except:
        print("Lựa chọn không hợp lệ, sử dụng Micro-Cosmos 1")
        test_name = "Micro-Cosmos 1"
        test_case = TEST_CASES[test_name]

    print(f"\n{'=' * 60}")
    if 'test_name' in locals():
        print(f"Test case: {test_name}")
    else:
        print(f"Test case: Custom test case")
    print(f"{'=' * 60}")

    # Parse test case
    player_pos, boxes, walls, goals = parse_test_case(test_case)

    if player_pos is None:
        print("Lỗi: Không tìm thấy người chơi (@) trong trạng thái ban đầu!")
        return

    if not boxes:
        print("Lỗi: Không có hộp ($) trong trạng thái ban đầu!")
        return

    if not goals:
        print("Lỗi: Không có đích (.) trong trạng thái ban đầu!")
        return

    initial_state = SokobanState(player_pos, boxes, walls, goals)

    # Hiển thị trạng thái ban đầu
    width = len(test_case[0])
    height = len(test_case)

    print("\nTrạng thái ban đầu:")
    print_state(initial_state, width, height)
    print(f"Số hộp: {len(boxes)}")
    print(f"Số đích: {len(goals)}")

    # Chọn thuật toán
    print("\nChọn thuật toán:")
    print("1. BFS (Breadth-First Search)")
    print("2. A* (A-star)")
    print("3. So sánh cả hai")

    algo_choice = input("Lựa chọn (1-3): ")

    results = {}

    if algo_choice in ['1', '3']:
        path, nodes, time_taken, frontier, visited = bfs_search(initial_state)
        if path:
            results['BFS'] = {
                'path': path,
                'length': len(path),
                'nodes': nodes,
                'time': time_taken,
                'frontier': frontier,
                'visited': visited,
                'test_name': test_name,
                'num_boxes': len(boxes),
                'table_size': f"{width}x{height}"
            }

    if algo_choice in ['2', '3']:
        path, nodes, time_taken, frontier, gscore_size = astar_search(initial_state)
        if path:
            results['A*'] = {
                'path': path,
                'length': len(path),
                'nodes': nodes,
                'time': time_taken,
                'frontier': frontier,
                'gscore_size': gscore_size
            }

    # Hiển thị kết quả
    if results:
        print("\n" + "=" * 60)
        print("KẾT QUẢ SO SÁNH")
        print("=" * 60)
        for algo_name, result in results.items():
            print(f"\n{algo_name}:")
            print(f"  - Độ dài lời giải: {result['length']} bước")
            print(f"  - Nodes đã duyệt: {result['nodes']}")
            print(f"  - Thời gian: {result['time']:.3f}s")
            print(f"  - Lời giải: {' '.join(result['path'])}")

        # So sánh hiệu suất
        if len(results) == 2:
            bfs_result = results['BFS']
            astar_result = results['A*']

            print(f"\n{'=' * 60}")
            print("PHÂN TÍCH SO SÁNH:")
            print(f"{'=' * 60}")

            speedup = bfs_result['time'] / astar_result['time']
            node_reduction = (1 - astar_result['nodes'] / bfs_result['nodes']) * 100

            print(f"- A* nhanh hơn BFS: {speedup:.2f}x")
            print(f"- A* giảm số nodes duyệt: {node_reduction:.1f}%")

            # Print comparison tables (time and memory) using helper
            print_comparison_tables(bfs_result, astar_result)

        # Demo solution
        demo = input("\nXem demo từng bước? (y/n): ")
        if demo.lower() == 'y':
            algo_demo = list(results.keys())[0]
            animate_solution(initial_state, results[algo_demo]['path'])
    else:
        print("\n✗ Không tìm thấy lời giải!")


if __name__ == "__main__":
    main()
