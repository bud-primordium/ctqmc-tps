from bridges import *
from plaquettes import *
import h5py

# 系统参数
N = 4
beta = 128.0
J = 1.0
num_sims = 10**5
checkpoint = 10**2
filesave = "observables.h5"

# 热退火的初始参数
initial_beta = 0.1
increment_small = 0.1
increment_large = 1.0
increment_change = 10.0
sims_increment = 1

def main():
    # 通过热退火进行预热
    Nx = N
    Ny = N
    traj = simulate_plaquettes(Nx, Ny, initial_beta)
    while traj.time <= beta:
        for _ in range(sims_increment * Nx * Ny):
            idx, pidxs = update_plaquette(traj, J)

        print("beta=", round(traj.time, 5))
        if traj.time == beta:
            break

        # 增加 beta
        inc = increment_large if traj.time + 1e-5 >= increment_change else increment_small
        traj.times *= (traj.time + inc) / traj.time
        traj.time += inc

    # 测量物理量
    Zs = [magnetization(traj, i) for i in range(1, Nx*Ny+1)]
    Xs = [len(traj.times[i]) for i in range(1, Nx*Ny+1)]

    # 更新
    X = 0
    Z = 0
    for i in range(1, num_sims * Nx * Ny + 1):
        # 更新平面
        idx, pidxs = update_plaquette(traj, J)

        # 测量物理量
        Xs[idx-1] = len(traj.times[idx])
        Zs[pidxs[0]-1] = magnetization(traj, pidxs[0])
        Zs[pidxs[1]-1] = magnetization(traj, pidxs[1])
        Zs[pidxs[2]-1] = magnetization(traj, pidxs[2])
        X += sum(Xs) / (Nx * Ny * beta)
        Z += sum(Zs) / (Nx * Ny * beta)

        # 保存结果
        if i % (checkpoint * Nx * Ny) == 0:
            print("sim=", i, "/", num_sims*Nx*Ny, ", Z=", round(Z / i, 5), ", X=", round(X / i, 5))
            with h5py.File(filesave, "w") as f:
                f.create_dataset("X", data=X / (N**2 * beta * i))
                f.create_dataset("Z", data=Z / (N**2 * beta * i))
                f.create_dataset("num_sims", data=i)

main()
