import numpy as np
import pygame
import sys
import time
from collections import deque

# 初始化 Pygame
pygame.init()
wind = pygame.display.set_mode([1000, 800])  # 視窗加寬，右側顯示統計數據
pygame.display.set_caption("BFS 迷宫求解 - 探索模式")
background = pygame.Surface(wind.get_size())
background.fill((255, 255, 255))

# 設置字體
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

# 生成固定地圖（障礙物不變）
def generate_fixed_board():
    board = np.ones((20, 20), dtype=int)  # 預設所有區域為可走區域 (1)
    
    # 隨機選擇障礙物 (0)
    num_obstacles = 50  
    random_indices = np.random.choice(board.size, num_obstacles, replace=False)
    np.put(board, random_indices, 0)

    # 隨機分配 2 和 3，代表不同移動權重
    num_special = 30  
    special_indices = np.random.choice(np.where(board.flatten() == 1)[0], num_special, replace=False)
    values = np.random.choice([2, 3], num_special, p=[0.5, 0.5])  
    np.put(board, special_indices, values)

    # 設置兩個起點：左下角與右下角
    start_positions = [[len(board)-1, 0], [len(board)-1, len(board[0])-1]]

    return board, start_positions

# 產生隨機終點（基於 1、2、3 權重，但終點未知）
def generate_weighted_goal(board):
    possible_positions = []
    weights = []
    
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] in [1, 2, 3]:  
                possible_positions.append([x, y])
                if board[x][y] == 1:
                    weights.append(1)
                elif board[x][y] == 2:
                    weights.append(2)
                elif board[x][y] == 3:
                    weights.append(3)

    goal_index = np.random.choice(len(possible_positions), p=np.array(weights) / sum(weights))
    return possible_positions[goal_index]

# 判斷是否合法
def isok(x, y, board):
    return 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] != 0

# BFS 探索 & 找到終點
def explore_bfs(start, board, hidden_goal):
    queue = deque([(start[0], start[1], [], 0)])  # (x, y, 路徑, 步數)
    visited = set()
    visited.add(tuple(start))

    discovered_goal = False  
    goal_position = None  

    while queue:
        x, y, path, steps = queue.popleft()
        
        # 模擬視野，當靠近終點時才會發現
        if not discovered_goal and (abs(x - hidden_goal[0]) + abs(y - hidden_goal[1]) <= 2):
            discovered_goal = True
            goal_position = hidden_goal  
        
        if discovered_goal and [x, y] == goal_position:
            return path + [[x, y]], goal_position, steps  

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            nx, ny = x + dx, y + dy
            if isok(nx, ny, board) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [[x, y]], steps + 1))
        
        # 即時顯示探索過的區域
        draw_map(board, goal_position, path, discovered=discovered_goal)
        time.sleep(0.1)  # 控制動畫速度

    return None, None, None  

# 繪製地圖
def draw_map(board, goal, explored_path=None, discovered=False, stats=None, last_steps=None):
    cell_size = 35  
    wind.fill((255, 255, 255))  

    # 繪製地圖
    for x in range(len(board)):
        for y in range(len(board[0])):
            rect_x, rect_y = y * cell_size, x * cell_size

            if board[x][y] == 1:
                color = (255, 155, 155)  # 粉色（普通區域）
            elif board[x][y] == 2:
                color = (255, 200, 100)  # 橙色（較容易生成終點）
            elif board[x][y] == 3:
                color = (255, 255, 100)  # 黃色（更容易生成終點）
            elif board[x][y] == 0:
                color = (255, 255, 255)  # 障礙物

            pygame.draw.rect(wind, color, (rect_x, rect_y, cell_size, cell_size))

    # 只有當終點被發現時才會顯示紅色
    if discovered:
        pygame.draw.rect(wind, (255, 0, 0), (goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size))

    # 繪製探索過的路徑（藍色）
    if explored_path:
        for i in range(1, len(explored_path)):  
            pygame.draw.line(wind, (0, 0, 255),  
                             (explored_path[i-1][1]*cell_size + cell_size//2, explored_path[i-1][0]*cell_size + cell_size//2),
                             (explored_path[i][1]*cell_size + cell_size//2, explored_path[i][0]*cell_size + cell_size//2), 3)

    # 顯示右上角 "上次步數"
    if last_steps is not None:
        step_text = font.render(f"上次步數: {last_steps}", True, (0, 0, 0))
        wind.blit(step_text, (800, 50))  # 右上角顯示

    # 繪製右側統計數據
    if stats:
        draw_statistics(stats)

    pygame.display.update()  

# 即時顯示統計數據
def draw_statistics(stats):
    x_offset = 720  # 右側顯示區域
    y_offset = 100

    text1 = font.render("步數統計", True, (0, 0, 0))
    wind.blit(text1, (x_offset, y_offset))
    y_offset += 30

    sorted_steps = sorted(stats.keys())

    for step in sorted_steps:
        text = font.render(f"{step} 步: {stats[step]} 次", True, (0, 0, 0))
        wind.blit(text, (x_offset, y_offset))
        y_offset += 30

# 測試模式
def test_mode(num_tests=100):
    step_counts = []
    stats = {}
    last_steps = 0  # 初始上次步數為 0

    for test_num in range(1, num_tests + 1):
        board, start_positions = generate_fixed_board()
        hidden_goal = generate_weighted_goal(board)

        for start in start_positions:
            explored_path, found_goal, steps = explore_bfs(start, board, hidden_goal)
            if steps is not None:
                step_counts.append(steps)
                last_steps = steps  # 更新上次步數

                # 更新統計數據
                if steps in stats:
                    stats[steps] += 1
                else:
                    stats[steps] = 1

            draw_map(board, hidden_goal, explored_path, discovered=True, stats=stats, last_steps=last_steps)
            time.sleep(0.3)  

            if found_goal:
                break  

# 執行測試模式
test_mode(100)

# Pygame 退出
pygame.quit()
