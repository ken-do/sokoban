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


def parse_level(level_map):
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

    for y, row in enumerate(level_map):
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
            return reconstruct_path(current_state), nodes_expanded, elapsed_time

        # Tạo các trạng thái kế tiếp
        for successor in get_successors(current_state):
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    elapsed_time = time.time() - start_time
    print(f"\n✗ Không tìm thấy lời giải sau {elapsed_time:.2f}s")
    return None, nodes_expanded, elapsed_time


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
            return reconstruct_path(current_state), nodes_expanded, elapsed_time

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
    return None, nodes_expanded, elapsed_time

def reconstruct_path(state):
    """Tái tạo đường đi từ trạng thái đích về trạng thái đầu"""
    path = []
    current = state
    while current.parent is not None:
        path.append(current.move)
        current = current.parent
    path.reverse()
    return path


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
    "Micro-Cosmos 1": [
        ['#', '#', '#', '#', '#'],
        ['#', '.', '@', ' ', '#'],
        ['#', '$', ' ', ' ', '#'],
        ['#', ' ', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#']
    ],

    "Micro-Cosmos 2": [
        ['#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', '$', '@', ' ', '#'],
        ['#', ' ', '.', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#']
    ],

    "Mini-Cosmos 1": [
        ['#', '#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', '.', '$', '.', ' ', '#'],
        ['#', ' ', '$', '.', '$', ' ', '#'],
        ['#', ' ', '.', '$', '.', ' ', '#'],
        ['#', ' ', ' ', '@', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', '#']
    ],

    "Medium 1": [
        ['#', '#', '#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', '.', '$', '$', '.', ' ', '#'],
        ['#', ' ', ' ', '.', '.', ' ', ' ', '#'],
        ['#', ' ', ' ', '@', ' ', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', '#', '#']
    ],

    "Simple Test": [
        ['#', '#', '#', '#', '#', '#'],
        ['#', '@', ' ', '$', '.', '#'],
        ['#', '#', '#', '#', '#', '#']
    ],

    "Two Boxes": [
        ['#', '#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', '$', '@', '$', ' ', '#'],
        ['#', ' ', '.', ' ', '.', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', '#']
    ],

    "Corner Challenge": [
        ['#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', '#', '#', ' ', '#'],
        ['#', '.', '#', '#', '$', '#'],
        ['#', ' ', '@', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#']
    ]
}

def main():
    print("=" * 60)
    print("SOKOBAN SOLVER - BFS & A*")
    print("=" * 60)

    # Chọn test case
    print("\nCác test case có sẵn:")
    test_names = list(TEST_CASES.keys())
    for i, name in enumerate(test_names, 1):
        print(f"{i}. {name}")

    print(f"{len(test_names) + 1}. Tự nhập level (custom)")

    choice = input(f"\nChọn test case (1-{len(test_names) + 1}): ")

    try:
        choice_num = int(choice)
        if choice_num == len(test_names) + 1:
            # Nhập custom level
            print("\nNhập level (mỗi dòng một hàng, nhấn Enter 2 lần để kết thúc):")
            print("Ký tự: # (tường), @ (player), $ (box), . (goal), * (box on goal)")
            level_map = []
            while True:
                line = input()
                if not line:
                    break
                level_map.append(list(line))
        else:
            test_name = test_names[choice_num - 1]
            level_map = TEST_CASES[test_name]
    except:
        print("Lựa chọn không hợp lệ, sử dụng Micro-Cosmos 1")
        test_name = "Micro-Cosmos 1"
        level_map = TEST_CASES[test_name]

    print(f"\n{'=' * 60}")
    if 'test_name' in locals():
        print(f"Test case: {test_name}")
    else:
        print(f"Test case: Custom Level")
    print(f"{'=' * 60}")

    # Parse level
    player_pos, boxes, walls, goals = parse_level(level_map)

    if player_pos is None:
        print("Lỗi: Không tìm thấy người chơi (@) trong level!")
        return

    if not boxes:
        print("Lỗi: Không có hộp ($) trong level!")
        return

    if not goals:
        print("Lỗi: Không có đích (.) trong level!")
        return

    initial_state = SokobanState(player_pos, boxes, walls, goals)

    # Hiển thị trạng thái ban đầu
    width = len(level_map[0])
    height = len(level_map)

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
        path, nodes, time_taken = bfs_search(initial_state)
        if path:
            results['BFS'] = {
                'path': path,
                'length': len(path),
                'nodes': nodes,
                'time': time_taken
            }

    if algo_choice in ['2', '3']:
        path, nodes, time_taken = astar_search(initial_state)
        if path:
            results['A*'] = {
                'path': path,
                'length': len(path),
                'nodes': nodes,
                'time': time_taken
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

        # Demo solution
        demo = input("\nXem demo từng bước? (y/n): ")
        if demo.lower() == 'y':
            algo_demo = list(results.keys())[0]
            animate_solution(initial_state, results[algo_demo]['path'])
    else:
        print("\n✗ Không tìm thấy lời giải!")


if __name__ == "__main__":
    main()
