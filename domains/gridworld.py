import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import OrderedDict


class GridWorld:
    """A class for making gridworlds"""
    # 设置动作，八个方向，与传统的上北下南不同
    ACTION = OrderedDict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))

    def __init__(self, image, target_x, target_y):
        self.image = image
        # 图片的行数和列数
        self.n_row = image.shape[0]
        self.n_col = image.shape[1]

        # 与建立障碍物地图时相反，等于0处是障碍物，不等于0处是自由空间
        self.obstacles = np.where(self.image == 0)
        self.freespace = np.where(self.image != 0)
        self.target_x = target_x
        self.target_y = target_y
        # 状态的个数，每一个格子表示一个状态
        self.n_states = self.n_row * self.n_col
        self.n_actions = len(self.ACTION)

        self.G, self.W, self.P, self.R, self.state_map_row, self.state_map_col = self.set_vals()

    # 取得了该位置在整个栅格地图中的索引值
    def loc_to_state(self, row, col):
        # (self.n_row, self.n_col)会构建一个n_row行n_col列的从0到n_row*n_col的数
        # F表示这些数优先按列排列，即第一列0到n_row-1.第二列n_row到n_row-1+n_row，以此类推
        # [row, col]是按照第row行第col列的索引取对应元素的值
        return np.ravel_multi_index([row, col], (self.n_row, self.n_col), order='F')

    def state_to_loc(self, state):
        return np.unravel_index(state, (self.n_col, self.n_row), order='F')

    def set_vals(self):
        # Setup function to initialize all necessary

        # Cost of each action, equivalent to the length of each vector
        #  i.e. [1., 1., 1., 1., 1.414, 1.414, 1.414, 1.414]
        action_values = self.ACTION.values()
        list_action_values = list(self.ACTION.values())
        # 平方和再开方
        action_cost = np.linalg.norm(list_action_values, axis=1)
        # Initializing reward function R: (curr_state, action) -> reward: float
        # Each transition has negative reward equivalent to the distance of transition
        # 每个状态的任何一个动作的奖赏，是移动距离的相反数
        R = - np.ones((self.n_states, self.n_actions)) * action_cost
        # Reward at target is zero
        target = self.loc_to_state(self.target_x, self.target_y)
        # target状态对应的8个方向奖赏都设为0，表示到了目的地点后往哪边走都不奖赏
        # 把target索引对应的那一行全部设为0
        R[target, :] = 0

        # Transition function P: (curr_state, next_state, action) -> probability: float
        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        # Filling in P
        for row in range(self.n_row):
            for col in range(self.n_col):
                curr_state = self.loc_to_state(row, col)
                for i_action, action in enumerate(self.ACTION):
                    # 每一个状态，即每个栅格，都向旁边8个方向走一次
                    neighbor_row, neighbor_col = self.move(row, col, action)
                    neighbor_state = self.loc_to_state(neighbor_row, neighbor_col)
                    # 凡是从当前栅格走到下一步栅格的,就置为1
                    # 用这种方式,把每个格子可能的动作全部走一遍,走完后所到的那个位置的值置为1
                    # 64个channel，每个channel有64行，8列
                    P[curr_state, neighbor_state, i_action] = 1

        # Adjacency matrix of a graph connecting curr_state and next_state
        # 对每channel的每行的所有列进行逻辑或
        # 表示从当前格子经过一步动作,不论何种动作,能否到达相邻格子
        G = np.logical_or.reduce(P, axis=2)
        # Weight of transition edges, equivalent to the cost of transition
        # 从当前格子到相邻格子的最大权重
        # 为什么不选最小呢?
        W = np.maximum.reduce(P * action_cost, axis=2)

        non_obstacles_unordered = self.loc_to_state(self.freespace[0], self.freespace[1])

        non_obstacles = np.sort(non_obstacles_unordered)

        # G[[a, b],:][:,[c, d]]表示先把G的a行b行全取出来,然后从取出来的结果中取c列和d列
        # 注意a，b，c，d如果给定，需要给定array，不能给定标量
        # G_non_obstacles会有non_obstacles行，non_obstacles列
        # 把没有障碍物的空间中每个格子走到相邻格子的 可行性， 最大权重， 是否走过， 奖赏 都摘出来
        G_non_obstacles = G[non_obstacles, :][:, non_obstacles]
        W_non_obstacles = W[non_obstacles, :][:, non_obstacles]
        P_non_obstacles = P[non_obstacles, :, :][:, non_obstacles, :]
        R_non_obstacles = R[non_obstacles, :]

        state_map_col, state_map_row = np.meshgrid(
            np.arange(0, self.n_col), np.arange(0, self.n_row))
        # 把非障碍物的格子的行列索引都取出来
        state_map_row = state_map_row.flatten('F')[non_obstacles]
        state_map_col = state_map_col.flatten('F')[non_obstacles]

        return G_non_obstacles, W_non_obstacles, P_non_obstacles, R_non_obstacles, state_map_row, state_map_col

    def get_graph(self):
        # Returns graph
        G = self.G
        W = self.W[self.W != 0]
        return G, W

    def get_graph_inv(self):
        # Returns transpose of graph
        G = self.G.T
        W = self.W.T
        return G, W

    def val_2_image(self, val):
        # Zeros for obstacles, val for free space
        im = np.zeros((self.n_row, self.n_col))
        im[self.freespace[0], self.freespace[1]] = val
        return im

    def get_value_prior(self):
        # Returns value prior for gridworld
        s_map_col, s_map_row = np.meshgrid(
            np.arange(0, self.n_col), np.arange(0, self.n_row))
        # 每个格子到目标点的欧几里得距离
        im = np.sqrt(
            np.square(s_map_col - self.target_y) +
            np.square(s_map_row - self.target_x))
        return im

    def get_reward_prior(self):
        # Returns reward prior for gridworld
        im = -1 * np.ones((self.n_row, self.n_col))
        # 目标点处的奖赏值设为10
        im[self.target_x, self.target_y] = 10
        return im

    def t_get_reward_prior(self):
        # Returns reward prior as needed for
        #  dataset generation
        im = np.zeros((self.n_row, self.n_col))
        im[self.target_x, self.target_y] = 10
        return im

    def get_state_image(self, row, col):
        # Zeros everywhere except [row,col]
        im = np.zeros((self.n_row, self.n_col))
        # 对应行列的值设为1
        im[row, col] = 1
        return im

    def map_ind_to_state(self, row, col):
        # Takes [row, col] and maps to a state
        rw = np.where(self.state_map_row == row)
        cl = np.where(self.state_map_col == col)
        return np.intersect1d(rw, cl)[0]

    def get_coords(self, states):
        # Given a state or states, returns
        #  [row,col] pairs for the state(s)
        non_obstacles = self.loc_to_state(self.freespace[0], self.freespace[1])
        non_obstacles = np.sort(non_obstacles)
        states = states.astype(int)
        r, c = self.state_to_loc(non_obstacles[states])
        return r, c

    def rand_choose(self, in_vec):
        # Samples
        if len(in_vec.shape) > 1:
            if in_vec.shape[1] == 1:
                in_vec = in_vec.T
        temp = np.hstack((np.zeros((1)), np.cumsum(in_vec))).astype('int')
        q = np.random.rand()
        x = np.where(q > temp[0:-1])
        y = np.where(q < temp[1:])
        return np.intersect1d(x, y)[0]

    def next_state_prob(self, s, a):
        # Gets next state probability for
        #  a given action (a)
        if hasattr(a, "__iter__"):
            p = np.squeeze(self.P[s, :, a])
        else:
            p = np.squeeze(self.P[s, :, a]).T
        return p

    def sample_next_state(self, s, a):
        # Gets the next state given the
        #  current state (s) and an
        #  action (a)
        vec = self.next_state_prob(s, a)
        result = self.rand_choose(vec)
        return result

    def get_size(self):
        # Returns domain size
        return self.n_row, self.n_col

    def move(self, row, col, action):
        # Returns new [row,col]
        #  if we take the action
        r_move, c_move = self.ACTION[action]
        # 不能走出第0行，也不能走出第n-1行
        new_row = max(0, min(row + r_move, self.n_row - 1))
        # 不能走出第0列，也不能走出第n-1列
        new_col = max(0, min(col + c_move, self.n_col - 1))
        # 移动一步后，如果img的值是0，表示此处有障碍物，不可移动，还将上一步的位置保持住
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


def trace_path(pred, source, target):
    # traces back shortest path from
    #  source to target given pred
    #  (a predicessor list)
    max_len = 1000
    path = np.zeros((max_len, 1))
    i = max_len - 1
    path[i] = target
    while path[i] != source and i > 0:
        try:
            path[i - 1] = pred[int(path[i])]
            i -= 1
        except Exception as e:
            return []
    if i >= 0:
        path = path[i:]
    else:
        path = None
    return path


def sample_trajectory(M: GridWorld, n_states):
    # Samples trajectories from random nodes
    #  in our domain (M)
    # 把栅格地图M的G和W取出
    G, W = M.get_graph_inv()
    N = G.shape[0]
    if N >= n_states:
        rand_ind = np.random.permutation(N)
    else:
        rand_ind = np.tile(np.random.permutation(N), (1, 10))
    # 用轨迹的个数，初始化这么多个起始点
    init_states = rand_ind[0:n_states].flatten()
    goal_s = M.map_ind_to_state(M.target_x, M.target_y)
    states = []
    states_xy = []
    states_one_hot = []
    # Get optimal path from graph
    g_dense = W
    # 把权重为0的全部屏蔽
    g_masked = np.ma.masked_values(g_dense, 0)
    g_sparse = csr_matrix(g_dense)
    # 使用迪杰斯特拉算法找到最短路径
    d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)
    # 对于每条轨迹都回溯，把所有轨迹的回溯结果存在一个array中
    for i in range(n_states):
        # 回溯得到从起点到终点的路径
        # 起始点可以是任意点，它是随机函数实现的
        path = trace_path(pred, goal_s, init_states[i])
        path = np.flip(path, 0)
        states.append(path)
    # 把路径上的每个点的行列索引
    for state in states:
        L = len(state)
        r, c = M.get_coords(state)
        row_m = np.zeros((L, M.n_row))
        col_m = np.zeros((L, M.n_col))
        for i in range(L):
            row_m[i, r[i]] = 1
            col_m[i, c[i]] = 1
        # one hot中有路径点的，设为1；没有路径点的，为0；
        states_one_hot.append(np.hstack((row_m, col_m)))
        states_xy.append(np.hstack((r, c)))
    return states_xy, states_one_hot
