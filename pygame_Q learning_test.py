import numpy as np
import pygame
import time
import random
from collections import deque

# 參數設置
GRID_SIZE = 20
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右
ALPHA = 0.1  # 學習率
GAMMA = 0.95  # 折扣因子
EPSILON = 0.05  # 探索機率
EPISODES = 200  # 訓練次數

# 生成地圖
def generate_fixed_board():
    board = np.array([
        [1, 1, 1, 0, 1, 2, 1, 1, 3, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1],
        [1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 1, 1],
        [1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1],
        [1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1],
        [1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1],
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 3],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int)
    return board

# 產生隨機終點
def generate_weighted_goal(board):
    possible_positions = np.argwhere(board != 0)
    return tuple(possible_positions[np.random.choice(len(possible_positions))])

# BFS 找最短路徑，初始化 Q-table
def bfs_initialize_qtable(board, start, goal, Q_table):
    queue = deque([(start, 0)])  # (當前位置, 路徑長度)
    visited = set()
    visited.add(start)
    while queue:
        (x, y), depth = queue.popleft()
        for action_idx, (dx, dy) in enumerate(ACTIONS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and board[nx, ny] != 0 and (nx, ny) not in visited:
                Q_table[x, y, action_idx] = 100 - depth  # 給最短路徑較高 Q 值
                visited.add((nx, ny))
                queue.append(((nx, ny), depth + 1))

# Q-table 初始化
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Q-learning 訓練
def train_q_learning():
    for episode in range(EPISODES):
        board = generate_fixed_board()
        start = (GRID_SIZE - 1, 0)
        goal = generate_weighted_goal(board)
        
        # 先用 BFS 初始化 Q-table
        bfs_initialize_qtable(board, start, goal, Q_table)
        
        state = start
        while state != goal:
            x, y = state
            
            if np.random.rand() < EPSILON:
                action_idx = np.random.choice(len(ACTIONS))
            else:
                action_idx = np.argmax(Q_table[x, y])
            
            dx, dy = ACTIONS[action_idx]
            next_state = (x + dx, y + dy)
            
            if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and board[next_state] != 0:
                reward = 100 if next_state == goal else -1
                Q_table[x, y, action_idx] = (1 - ALPHA) * Q_table[x, y, action_idx] + ALPHA * (reward + GAMMA * np.max(Q_table[next_state]))
                state = next_state
            else:
                Q_table[x, y, action_idx] -= 10  # 撞牆懲罰
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Training in progress...")

# 測試並顯示結果
def test_q_learning():
    pygame.init()
    wind = pygame.display.set_mode([700, 700])
    pygame.display.set_caption("Q-learning 迷宫 AI")
    wind.fill((255, 255, 255))
    cell_size = 35

    board = generate_fixed_board()
    start = (GRID_SIZE - 1, 0)
    goal = generate_weighted_goal(board)
    state = start
    path = [state]

    while state != goal:
        x, y = state
        action_idx = np.argmax(Q_table[x, y])
        dx, dy = ACTIONS[action_idx]
        next_state = (x + dx, y + dy)

        if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE and board[next_state] != 0:
            state = next_state
            path.append(state)
        else:
            # 懲罰錯誤選擇並重新挑選行動
            Q_table[x, y, action_idx] = -1000
            action_idx = np.argmax(Q_table[x, y])  

    # 繪製地圖
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            color = (255, 255, 255) if board[x, y] == 0 else (200, 200, 200)
            pygame.draw.rect(wind, color, (y * cell_size, x * cell_size, cell_size, cell_size))
    pygame.draw.rect(wind, (255, 0, 0), (goal[1] * cell_size, goal[0] * cell_size, cell_size, cell_size))

    # 繪製 AI 走過的路徑
    for i in range(1, len(path)):  
        pygame.draw.line(wind, (0, 0, 255),  
                         (path[i-1][1]*cell_size + cell_size//2, path[i-1][0]*cell_size + cell_size//2),
                         (path[i][1]*cell_size + cell_size//2, path[i][0]*cell_size + cell_size//2), 3)
        pygame.display.update()
        time.sleep(0.1)

    pygame.quit()


# 訓練與測試
train_q_learning()
test_q_learning()
