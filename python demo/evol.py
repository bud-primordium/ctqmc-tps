import numpy as np

# 处理从状态si开始，经过时间t，最后在状态sf结束的自旋随机桥的计算和生成。
# 其中，可以选择由某种偏置强度s给出的磁化偏置。

def partition(J, t, i, f):
    """
    确定从i到f在时间t的桥的对数分区函数和z-磁化J。
    """
    if i == True and f == True:
        Z = t * np.sqrt(1 + J**2) - np.log(2)
        Z += np.log(1 + (J / np.sqrt(1 + J**2)) + np.exp(-2 * t * np.sqrt(1 + J**2)) * (1 - (J / np.sqrt(1 + J**2))))
    elif i == False and f == False:
        Z = t * np.sqrt(1 + J**2) - np.log(2)
        Z += np.log(1 - (J / np.sqrt(1 + J**2)) + np.exp(-2 * t * np.sqrt(1 + J**2)) * (1 + (J / np.sqrt(1 + J**2))))
    else:
        Z = t * np.sqrt(1 + J**2) - np.log(2) - np.log(np.sqrt(1 + J**2))
        Z += np.log(1 -  np.exp(-2 * t * np.sqrt(1 + J**2)))
    return Z


def survival(J, t, tau, i, f):
    """
    确定在时间tau的生存概率，从某个初始状态i和最终状态f，具有z-磁化J和总时间t。
    """
    return np.exp(partition(J, t - tau, i, f) - partition(J, t, i, f) + tau * (J if i else -J))


def survival_time(J, t, i, f):
    """
    计算具有磁化J，时间t，初始状态i和最终状态f的系统的生存时间。
    """
    # 生成一个随机数，决定时间
    r = np.random.random()

    # 初始化时间限制
    lower = 0
    upper = t

    # 计算完整时间的分区和
    Z = partition(J, t, i, f)

    # 检查状态是否存活
    if i == f:
        S = np.exp(partition(J, 0, i, f) - Z +  t * (J if i else -J))
        if S > r: 
            return t + 1e-5

    # 二分查找，直到找到目标值
    tau = 0.5 * (upper + lower)
    val = np.exp(partition(J, t - tau, i, f) - Z + tau * (J if i else -J))
    while abs(val - r) > 1e-8 or np.isnan(val):
        if val > r and not np.isnan(val):
            lower = 0.5 * (upper + lower)
        else:
            upper = 0.5 * (upper + lower)
        tau = 0.5 * (upper + lower)
        val = np.exp(partition(J, t - tau, i, f) - Z + tau * (J if i else -J))

    return tau


def bridge(J, t, i, f):
    """
    采样从初始状态i到最终状态f的桥，时间为t，耦合为J。
    """
    # 初始化时间
    time = 0.0
    times = []

    # 找出所有转换时间
    while time < t:
        # 确定生存时间
        dt = survival_time(J, t - time, i, f)

        # 更新系统
        if time + dt < t:
            # 存储跳跃
            time += dt
            times.append(time)
            i = not i
        else:
            # 生存
            time += dt
    return times