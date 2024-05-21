include("evolution.jl")
include("triangle_lattice.jl")
using HDF5
"""
HDF5是真的好用woc
妈妈再也不用担心我的
csv/txt写一半电脑罢工了
"""

# 系统属性
N = 4
beta = 128.0 #遵循原论文，我们演化到逆温度为128再开始计算
num_sims = 10^4 

"""
10^5会准一些，但电脑有点难承受，经济出发咱们10^4就够了，个别点感兴趣可以试试10^5
调到10^5我的laptop就得半小时一个数据点
"""
checkpoint = 100
#filesave = "result at 0.0-0.8.h5"
#filesave = "result at 0.9-1.1-morenums.h5"
filesave = "result at 1.2-2.0.h5"

# 热退火属性
initial_beta = 0.1
increment_small = 0.1
increment_large = 1.0
increment_change = 10.0
sims_increment = 1

function anneal_system!(traj, J, Nx, Ny, beta)
    while traj.time <= beta
        for _ in 1:sims_increment * Nx * Ny
            updatePlaquette!(traj, J)
        end

        println("逆温度beta=$(round(traj.time, digits=5))")
        if traj.time == beta
            break
        end

        # 增加 beta
        inc = (traj.time + 1e-5) >= increment_change ? increment_large : increment_small
        traj.times .*= (traj.time + inc) / traj.time
        traj.time += inc
    end
end
using HDF5

function run_simulation(J, Nx, Ny, beta, sim_id)
    traj = simulatePlaquettes(Nx, Ny, initial_beta)
    anneal_system!(traj, J, Nx, Ny, beta)

    Zs = [magnetization(traj, i) for i in 1:Nx * Ny]
    Xs = [length(traj.times[i]) for i in 1:Nx * Ny]

    X, Z, Z_var = 0.0, 0.0, 0.0
    for step in 1:num_sims * Nx * Ny
        idx, pidxs = updatePlaquette!(traj, J)

        Xs[idx] = length(traj.times[idx])
        for pidx in pidxs
            Zs[pidx] = magnetization(traj, pidx)
        end
        X += sum(Xs) / (Nx * Ny * beta)
        Z += sum(Zs) / (Nx * Ny * beta)

        Zs_mean = sum(Zs) / (Nx * Ny)
        Zs_var = sum((Zs .- Zs_mean).^2) / (Nx * Ny)
        Z_var += Zs_var / (Nx * Ny * beta)

        if step % (checkpoint * Nx * Ny) == 0
            println("step=$step/$(num_sims * Nx * Ny), Z=$(round(Z / step, digits=5)), X=$(round(X / step, digits=5)), Z_var=$(round(Z_var * beta / step, digits=5)), J=$J")
            
            # 在HDF5文件中创建一个新的组，用于存储当前检查点的数据
            h5write(filesave, "sim_$sim_id/step_$step/X", X / step)
            h5write(filesave, "sim_$sim_id/step_$step/Z", Z / step)
            h5write(filesave, "sim_$sim_id/step_$step/Z_var", Z_var * beta / step)
            h5write(filesave, "sim_$sim_id/step_$step/num_sims", step)
            h5write(filesave, "sim_$sim_id/step_$step/J", J)
        end
    end
end

function main()
    Nx, Ny = N, N

    sim_id = 1  # 初始化模拟ID
    #for J in 0.0:0.1:0.8
    #for J in 0.9:0.01:1.1
    for J in 1.2:0.1:2.0
    """
    这里分三段模拟，注意上面的文件名也要改
    """
        println("现在模拟的是 J = $J")
        run_simulation(J, Nx, Ny, beta, sim_id)
        sim_id += 1  # 每次模拟后增加模拟ID
    end
end

main()

