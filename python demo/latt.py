import numpy as np
import random
from bridges import bridge
from typing import List


# 定义 Plaquettes 结构体
class Plaquettes:
    """
    存储 plaquette 轨迹的结构体。
    """
    def __init__(self, Nx: int, Ny: int, time: float, initial: np.ndarray, times: List[np.ndarray]):
        self.Nx = Nx
        self.Ny = Ny
        self.time = time
        self.initial = initial
        self.times = times

# 定义 Plaquettes 函数
def create_plaquettes(Nx: int, Ny: int, time: float, initial: np.ndarray) -> Plaquettes:
    """
    创建一个 Plaquettes 对象。

    参数:
    - Nx (int): 格子中的列数。
    - Ny (int): 格子中的行数。
    - time (float): 总的模拟时间。
    - initial (np.ndarray): 格子的初始状态。

    返回:
    - Plaquettes: 一个 Plaquettes 对象。
    """
    times = [np.array([]) for _ in range(Nx*Ny)]
    return Plaquettes(Nx, Ny, time, initial, times)

def simulate_plaquettes(Nx, Ny, time):
    initial = np.random.choice([False, True], size=Nx*Ny)
    times = [bridge(0, time, False, False) for _ in range(Nx*Ny)]
    return Plaquettes(Nx, Ny, time, initial, times)

def plaquette_position(P, idx):
    # 确保索引在范围内
    idx = idx if idx <= (P.Nx * P.Ny) else idx - (P.Nx * P.Ny)
    idx = idx if idx >= 1 else (P.Nx * P.Ny) + idx

    # 查找 plaquette 的坐标
    ypos = int(np.ceil(idx / P.Nx))
    xpos = idx - ((ypos - 1) * P.Nx)

    return xpos, ypos

def plaquette_index(P, xpos, ypos):
    # 确保位置在范围内
    xpos = xpos if xpos <= P.Nx else xpos - P.Nx
    xpos = xpos if xpos >= 1 else P.Nx + xpos
    ypos = ypos if ypos <= P.Ny else ypos - P.Ny
    ypos = ypos if ypos >= 1 else P.Ny + ypos

    # 转换为索引
    return (ypos - 1) * P.Nx + xpos

def spin_plaquette_indexes(P, sidx):
    """
    返回与索引为sidx的自旋相关的平面坐标。
    """
    # 找到第一个站点的坐标
    xpos, ypos = plaquette_position(P, sidx)

    # 找到其他平面的索引
    idx2 = plaquette_index(P, xpos, ypos+1)
    idx3 = plaquette_index(P, xpos+1, ypos+1)

    return sidx, idx2, idx3

def plaquette_spin_indexes(P, idx):
    """
    返回包含在一个平面中的自旋的索引。
    """
    # 找到平面索引的坐标
    xpos, ypos = plaquette_position(P, idx)

    # 找到其他自旋的索引
    sidx2 = plaquette_index(P, xpos, ypos-1)
    sidx3 = plaquette_index(P, xpos-1, ypos-1)

    return idx, sidx2, sidx3

def update_plaquette(P, J):
    """
    更新一个平面。
    """
    # 选择一个索引
    i = random.randint(1, P.Nx*P.Ny)

    # 采样并更新轨迹
    pidxs, initial, times = sample_plaquette(P, i, J)
    P.initial[pidxs] = initial 
    P.times[i] = times

    return i, pidxs

def sample_plaquette(P, sidx, J):
    """
    采样一个平面的新轨迹。
    """
    # 找到包含该自旋的平面
    pidxs = spin_plaquette_indexes(P, sidx)
    
    # 找到其他翻转邻近自旋的重构
    times, transitions = reconstruct_spin(P, sidx)

    # 创建可能的平面状态列表及其 Z-mags
    states = [[1, 1, 1], [0, 0, 0],
              [0, 1, 1], [1, 0, 0],
              [1, 0, 1], [0, 1, 0],
              [1, 1, 0], [0, 0, 1]]
    Zs = J * [3, 1, 1, 1]

    # 进化算子（调整归一化以避免过大）
    def evolution(t):
        U = np.zeros((8, 8), dtype=np.float64)

        # 第一个矩阵
        Jprime = np.sqrt(1 + Zs[0]**2)
        ex = np.exp(-2 * t * Jprime)
        U[0, 0] = 0.5 * (1 + (Zs[0] / Jprime) + ex * (1 - (Zs[0] / Jprime)))
        U[1, 1] = 0.5 * (1 - (Zs[0] / Jprime) + ex * (1 + (Zs[0] / Jprime)))
        U[0, 1] = (0.5 / Jprime) * (1 - ex)
        U[1, 0] = U[0, 1]

        # 第二个矩阵
        Jprime2 = np.sqrt(1 + Zs[1]**2)
        ex = np.exp(-2 * t * Jprime2)
        ex2 = np.exp(t * (Jprime2 - Jprime))
        U[2, 2] = 0.5 * ex2 * (1 + (Zs[1] / Jprime2) + ex * (1 - (Zs[1] / Jprime2)))
        U[3, 3] = 0.5 * ex2 * (1 - (Zs[1] / Jprime2) + ex * (1 + (Zs[1] / Jprime2)))
        U[2, 3] = (0.5 / Jprime2) * ex2 * (1 - ex)
        U[3, 2] = U[2, 3]

        # 重复
        U[4:5, 4:5] = U[2:3, 2:3]
        U[6:7, 6:7] = U[2:3, 2:3]
        return U

    # 创建翻转矩阵
    flips = np.zeros((3, 8, 8), dtype=np.float64)
    idxs = [[2, 3, 0, 1, 7, 6, 5, 4],
            [4, 5, 7, 6, 0, 1, 3, 2],
            [6, 7, 5, 4, 3, 2, 0, 1]]
    for i in range(3):
        for j in range(8):
            flips[i, j, idxs[i][j]] = 1.0

    # 创建单位矩阵
    Q = np.eye(8)

    Ps = np.zeros((len(times), 8, 8), dtype=np.float64)
    Us = np.zeros((len(times)+1, 8, 8), dtype=np.float64)
    # 进化矩阵
    for i, t in enumerate(times):
        dt = t if i == 0 else t - times[i-1]
        U = evolution(dt)
        idx = pidxs.index(transitions[i])
        U = np.dot(flips[idx], U)
        Us[i] = U
        Q = np.dot(U, Q)
        Q /= np.sum(Q)

        Ps[i] = np.copy(Q)

    # 最后一次进化
    dt = P.time - times[-1] if len(times) > 0 else P.time
    U = evolution(dt)
    Us[-1] = U
    Q = np.dot(U, Q)
    Q /= np.sum(np.diag(Q))

    # 采样初始/最终配置
    configs = np.zeros(len(times)+2, dtype=np.int)
    idx = np.argmax(np.cumsum(np.diag(Q)) >= np.random.rand())
    configs[0] = idx 
    configs[-1] = idx

    # 反向采样配置
    for i in range(len(times)):
        Q = Ps[-i-1, :, configs[0]]
        U = Us[-i-1, configs[i], :]
        Q = U * Q 
        Q /= np.sum(Q)

        idx = np.argmax(np.cumsum(Q) >= np.random.rand())
        configs[i+1] = idx

    # 强制转换前的配置
    configs_ends = np.zeros(len(times)+1, dtype=np.int)
    for i in range(len(times)):
        idx = configs[i+1]
        state = states[idx].copy()
        idx = pidxs.index(transitions[i])
        state[idx] = 1 - state[idx]
        idx = np.argmax([np.array_equal(states[j], state) for j in range(8)])
        configs_ends[i] = idx
    configs_ends[-1] = configs[-1]
    
    # 采样 bridges
    new_times = []
    for i in range(len(configs_ends)):
        J = Zs[int(np.ceil(configs[i] / 2))]
        tmin = 0.0 if i == 0 else times[i-1]
        tmax = P.time if i == len(configs_ends)-1 else times[i]
        bridge_times = bridge(J, tmax-tmin, configs[i] % 2 == 1, configs_ends[i] % 2 == 1)
        new_times.extend(bridge_times + tmin)
    
    return pidxs, states[configs[0]], new_times

def reconstruct_spin(P, sidx):
    """
    确定自旋的轨迹。
    """
    # 找到包含该自旋的平面
    pidxs = spin_plaquette_indexes(P, sidx)

    # 找到与平面关联的其他自旋
    sidxs = []
    for pidx in pidxs:
        idxs = plaquette_spin_indexes(P, pidx)
        for idx in idxs:
            if idx != sidx:
                sidxs.append(idx)
    pidxs = [pidxs[0], pidxs[0], pidxs[1], pidxs[1], pidxs[2], pidxs[2]]

    # 找到自旋翻转的时间列表
    next_times = [P.times[idx][0] if len(P.times[idx]) > 0 else P.time for idx in sidxs]
    next_idxs = [1 for _ in range(len(sidxs))]
    num_jumps = sum([len(P.times[idx]) for idx in sidxs])

    # 构建轨迹
    times = np.zeros(num_jumps)
    states = np.zeros(num_jumps)
    for i in range(num_jumps):
        idx = np.argmin(next_times)
        times[i] = next_times[idx]
        next_idxs[idx] += 1
        next_times[idx] = P.times[sidxs[idx]][next_idxs[idx]-1] if next_idxs[idx]-1 < len(P.times[sidxs[idx]]) else P.time
        states[i] = pidxs[idx]

    return times, states

def reconstruct_plaquette(P, idx):
    """
    确定平面自旋的轨迹。
    """
    # 找到平面中的自旋
    sidxs = plaquette_spin_indexes(P, idx)

    # 找到自旋翻转的时间列表
    next_times = [P.times[idx][0] if len(P.times[idx]) > 0 else P.time for idx in sidxs]
    next_idxs = [1 for _ in range(len(sidxs))]
    num_jumps = sum([len(P.times[idx]) for idx in sidxs])

    # 构建轨迹
    times = np.zeros(num_jumps+1)
    for i in range(num_jumps):
        idx = np.argmin(next_times)
        times[i+1] = next_times[idx]
        next_idxs[idx] += 1
        next_times[idx] = P.times[sidxs[idx]][next_idxs[idx]-1] if next_idxs[idx]-1 < len(P.times[sidxs[idx]]) else P.time

    return times

def magnetization(P, idx):
    """
    计算平面的磁化。
    """
    # 找到翻转的时间重构
    times = reconstruct_plaquette(P, idx)

    # 时间积分
    Z = 0
    state = P.initial[idx].copy()
    for i in range(len(times)):
        tmin = times[i]
        tmax = P.time if i == len(times)-1 else times[i+1]
        Z += (tmax - tmin) * (2 * state - 1)
        state = np.logical_not(state)
    
    return Z
