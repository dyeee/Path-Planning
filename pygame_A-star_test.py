import numpy as np
import pygame
import sys
import time
import heapq  # 用於 A* 演算法

# 初始化 Pygame
pygame.init()
wind = pygame.display.set_mode([800, 800])  
pygame.display.set_caption("A* 迷宫求解 - 探索模式")
background = pygame.Surface(wind.get_size())
background.fill((255, 255, 255))

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

# 啟發式函數（曼哈頓距離）
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* 搜索
def a_star_search(start, board, hidden_goal):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    open_set = []
    heapq.heappush(open_set, (0, 0, start[0], start[1], []))  # (總成本, G成本, x, y, 路徑)
    visited = set()
    discovered_goal = False  
    goal_position = None  

    while open_set:
        _, g_cost, x, y, path = heapq.heappop(open_set)

        # 如果已經到達目標點
        if discovered_goal and [x, y] == goal_position:
            return path + [[x, y]], goal_position  

        # 進入視野，發現目標點
        if not discovered_goal and (abs(x - hidden_goal[0]) + abs(y - hidden_goal[1]) <= 2):
            discovered_goal = True
            goal_position = hidden_goal  

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and board[nx][ny] != 0:
                new_cost = g_cost + 1  # 移動成本
                if (nx, ny) in path:
                    new_cost += 2  # 回頭路額外扣 2 分

                f_cost = new_cost + heuristic((nx, ny), hidden_goal)
                heapq.heappush(open_set, (f_cost, new_cost, nx, ny, path + [[x, y]]))
                visited.add((nx, ny))

        # 即時顯示探索過的區域
        draw_map(board, goal_position, path, discovered=discovered_goal)
        time.sleep(0.1)  # 控制動畫速度

    return None, None  

# 繪製地圖
def draw_map(board, goal, explored_path=None, discovered=False):
    cell_size = 35  
    wind.fill((255, 255, 255))  

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

    pygame.display.update()  

# 測試模式
def test_mode(num_tests=10):
    for _ in range(num_tests):
        board, start_positions = generate_fixed_board()
        hidden_goal = generate_weighted_goal(board)

        discovered_goal = False  
        goal_position = None  

        for start in start_positions:
            explored_path, found_goal = a_star_search(start, board, hidden_goal)
            
            if found_goal:  
                discovered_goal = True
                goal_position = found_goal  

            draw_map(board, goal_position, explored_path, discovered=discovered_goal)
            time.sleep(0.3)  

            if discovered_goal:  
                break  

        pygame.event.pump()  
        time.sleep(0.3)  

# 執行測試模式
test_mode(10)

# Pygame 退出
pygame.quit()
