#=
    根据原文精神，实现一个镶嵌格子（三角小板子Plaquettes）的类，该类存储了镶嵌格子的轨迹，
    以及用于模拟/操作它们的函数。
=#

using LinearAlgebra

"""
    Plaquettes(Nx::Int, Ny::Int, time::Real, initial::Array{Bool}, times::Array{Vector{Real}}})

存储镶嵌格子的轨迹。
"""
mutable struct Plaquettes{T<:Real}
    Nx::Int 
    Ny::Int
    time::T
    initial::Array{Bool}
    times::Array{Vector{T}}
end

function Plaquettes(Nx::Int, Ny::Int, time::Real, initial::Array{Bool})
    return Plaquettes(Nx, Ny, time, initial, [Float64[] for i = 1:Nx*Ny])
end

function simulatePlaquettes(Nx::Int, Ny::Int, time::Real)
    initial = rand(Bool, Nx*Ny)
    times = [bridge(0, time, false, false) for _ in 1:Nx*Ny]
    return Plaquettes(Nx, Ny, time, initial, times)
end

"""
    plaquettePosition(P::Plaquettes, idx::Int)

找到镶嵌的 x 和 y 坐标。
"""
function plaquettePosition(P::Plaquettes, idx::Int)
    # 确保索引在范围内
    idx = idx > (P.Nx * P.Ny) ? idx - (P.Nx * P.Ny) : idx
    idx = idx < 1 ? (P.Nx * P.Ny) + idx : idx

    # 找到镶嵌的坐标
    ypos = Int(ceil(idx / P.Nx))
    xpos = idx - ((ypos - 1) * P.Nx)

    return xpos, ypos
end

"""
    plaquetteIndex(P::Plaquettes, xpos::Int, ypos::Int)

给定镶嵌的位置，计算它的索引。
"""
function plaquetteIndex(P::Plaquettes, xpos::Int, ypos::Int)
    # 确保位置在范围内
    xpos = xpos > P.Nx ? xpos - P.Nx : xpos
    xpos = xpos < 1 ? P.Nx + xpos : xpos 
    ypos = ypos > P.Ny ? ypos - P.Ny : ypos
    ypos = ypos < 1 ? P.Ny + ypos : ypos 

    # 转换为索引
    return (ypos - 1) * P.Nx + xpos
end

"""
    spinPlaquetteIndexs(P::Plaquettes, sidx::Int)

返回与索引 sidx 处的自旋相关的镶嵌的索引。
"""
function spinPlaquetteIndexs(P::Plaquettes, sidx::Int)
    # 找到第一个位置的坐标
    xpos, ypos = plaquettePosition(P, sidx)

    # 找到其他镶嵌的索引
    idx2 = plaquetteIndex(P, xpos, ypos + 1)
    idx3 = plaquetteIndex(P, xpos + 1, ypos + 1)

    return sidx, idx2, idx3    
end

"""
    plaquetteSpinIndexs(P::Plaquettes, idx::Int)

返回镶嵌中包含的自旋的索引。
"""
function plaquetteSpinIndexs(P::Plaquettes, idx::Int)
    # 找到镶嵌索引的坐标
    xpos, ypos = plaquettePosition(P, idx)

    # 找到其他自旋的索引
    sidx2 = plaquetteIndex(P, xpos, ypos - 1)
    sidx3 = plaquetteIndex(P, xpos - 1, ypos - 1)

    return idx, sidx2, sidx3
end

"""
    updatePlaquette!(P::Plaquettes, J::Real)

更新一个镶嵌。
"""
function updatePlaquette!(P::Plaquettes, J::Real)
    # 选择一个索引
    i = rand(1:P.Nx * P.Ny)

    # 采样并更新轨迹
    pidxs, initial, times = samplePlaquette(P, i, J)
    P.initial[[pidxs...]] = initial 
    P.times[i] = times

    return i, pidxs
end

"""
    samplePlaquette(P::Plaquettes, sidx::Int, J::Real)  

为一个镶嵌采样一个新的轨迹。
"""
function samplePlaquette(P::Plaquettes, sidx::Int, J::Real)
    # 找到包含该自旋的镶嵌
    pidxs = spinPlaquetteIndexs(P, sidx)
    
    # 找到翻转其他相邻自旋的重构
    times, transitions = reconstructSpin(P, sidx)

    # 创建可能的镶嵌状态及其 Z-mags 列表
    states = [[1, 1, 1], [0, 0, 0],
              [0, 1, 1], [1, 0, 0],
              [1, 0, 1], [0, 1, 0],
              [1, 1, 0], [0, 0, 1]]
    Zs = J .* [3, 1, 1, 1]

    # 演化算符（归一化调整不使其太大）
    function evolution(t::Real)
        U = zeros(Float64, 8, 8)

        # 第一个矩阵
        Jprime = sqrt(1 + Zs[1]^2)
        ex = exp(-2 * t * Jprime)
        U[1, 1] = 0.5 * (1 + (Zs[1] / Jprime) + ex * (1 - (Zs[1] / Jprime)))
        U[2, 2] = 0.5 * (1 - (Zs[1] / Jprime) + ex * (1 + (Zs[1] / Jprime)))
        U[1, 2] = (0.5 / Jprime) * (1 - ex)
        U[2, 1] = U[1, 2]

        # 第二个矩阵
        Jprime2 = sqrt(1 + Zs[2]^2)
        ex = exp(-2 * t * Jprime2)
        ex2 = exp(t * (Jprime2 - Jprime))
        U[3, 3] = 0.5 * ex2 * (1 + (Zs[2] / Jprime2) + ex * (1 - (Zs[2] / Jprime2)))
        U[4, 4] = 0.5 * ex2 * (1 - (Zs[2] / Jprime2) + ex * (1 + (Zs[2] / Jprime2)))
        U[3, 4] = (0.5 / Jprime2) * ex2 * (1 - ex)
        U[4, 3] = U[3, 4]

        # 重复
        U[5:6, 5:6] = U[3:4, 3:4]
        U[7:8, 7:8] = U[3:4, 3:4]
        return U
    end

    # 创建翻转矩阵
    flips = zeros(Float64, 3, 8, 8)
    idxs = [[3, 4, 1, 2, 8, 7, 6, 5],
            [5, 6, 8, 7, 1, 2, 4, 3],
            [7, 8, 6, 5, 4, 3, 1, 2]]
    for i in 1:3
        for j in 1:8
            flips[i, j, idxs[i][j]] = 1.0
        end
    end
    
    # 创建单位矩阵
    Q = diagm(ones(Float64, 8))

    Ps = zeros(Float64, length(times), 8, 8)
    Us = zeros(Float64, length(times) + 1, 8, 8)
    # 演化矩阵
    for i in 1:length(times)
        # 找到演化矩阵
        dt = i == 1 ? times[1] : times[i] - times[i-1]
        U = evolution(dt)

        # 执行所需的翻转
        idx = findfirst(pidxs .== transitions[i])
        U = flips[idx, :, :] * U
        Us[i, :, :] = U

        # 演化
        Q = U * Q
        
        # 归一化
        Q ./= sum(Q)
        Ps[i, :, :] = deepcopy(Q)
    end
    
    # 做最终的演化
    dt = length(times) > 0 ? P.time - times[end] : P.time
    U = evolution(dt)
    Us[end, :, :] = U
    Q = U * Q
    Q ./= sum(Q[i, i] for i in 1:8)

    # 采样初始/最终配置
    configs = zeros(Int, length(times) + 2)
    idx = findfirst(cumsum(diag(Q)) .>= rand())
    configs[1] = idx 
    configs[end] = idx

    # 反向采样配置
    for i in 1:length(times)
        # 获取所需时间之前的演化
        Q = Ps[end-i+1, :, configs[1]]

        # 获取下一个演化矩阵
        U = Us[end-i+1, configs[end-i+1], :]

        # 找到概率
        Q = U .* Q 
        Q ./= sum(Q)

        # 采样
        idx = findfirst(cumsum(Q) .>= rand())
        configs[end-i] = deepcopy(idx) 
    end

    # 找到强制转换之前的配置
    configs_ends = zeros(Int, length(times) + 1)
    for i in 1:length(times)
        idx = configs[i + 1]
        state = deepcopy(states[idx])
        idx = findfirst(pidxs .== transitions[i])
        state[idx] = 1 - state[idx]
        idx = findfirst([states[j] == state for j in 1:8])
        configs_ends[i] = idx
    end
    configs_ends[end] = configs[end]
    
    # 采样桥
    new_times = Float64[]
    for i in 1:length(configs_ends)
        # 找到 J 值和时间
        J = Zs[Int(ceil(configs[i] / 2))]
        tmin = i == 1 ? 0.0 : times[i-1]
        tmax = i == length(configs_ends) ? P.time : times[i]
        bridge_times = bridge(J, tmax - tmin, isodd(configs[i]), isodd(configs_ends[i]))
        append!(new_times, bridge_times .+ tmin)
    end
    
    return pidxs, states[configs[1]], new_times
end


"""
    reconstructSpin(P::Plaquettes, sidx::Int)

确定一个自旋的轨迹。
"""
function reconstructSpin(P::Plaquettes, sidx::Int)
    # 找到包含该自旋的镶嵌
    pidxs = spinPlaquetteIndexs(P, sidx)

    # 找到与镶嵌相关的其他自旋
    sidxs = zeros(Int, 6) # 存储影响给定自旋索引的自旋索引
    i = 1
    for pidx in pidxs 
        idxs = plaquetteSpinIndexs(P, pidx)
        for idx in idxs
            if idx != sidx 
                sidxs[i] = idx 
                i += 1
            end
        end
    end
    pidxs = [pidxs[1], pidxs[1], pidxs[2], pidxs[2], pidxs[3], pidxs[3]]

    # 找到自旋翻转的时间列表
    next_times = [length(P.times[idx]) > 0 ? P.times[idx][1] : P.time for idx in sidxs]
    next_idxs = [1 for _ in 1:length(sidxs)]
    num_jumps = sum(length(P.times[idx]) for idx in sidxs)

    # 构建轨迹
    times = zeros(Float64, num_jumps)
    states = zeros(Int, num_jumps)
    for i in 1:num_jumps
        # 找到下一个时间
        idx = argmin(next_times)
        times[i] = next_times[idx]

        # 更新下一个时间列表
        next_idxs[idx] += 1
        next_times[idx] = next_idxs[idx] > length(P.times[sidxs[idx]]) ? P.time : P.times[sidxs[idx]][next_idxs[idx]]

        # 更新状态
        states[i] = pidxs[idx]
    end

    return times, states
end


"""
    reconstructPlaquette(P::Plaquettes, idx::Int)

确定一个镶嵌的自旋轨迹。
"""
function reconstructPlaquette(P::Plaquettes, idx::Int)
    # 找到包含该镶嵌的自旋
    sidxs = plaquetteSpinIndexs(P, idx)

    # 找到自旋翻转的时间列表
    next_times = [length(P.times[idx]) > 0 ? P.times[idx][1] : P.time for idx in sidxs]
    next_idxs = [1 for _ in 1:length(sidxs)]
    num_jumps = sum(length(P.times[idx]) for idx in sidxs)

    # 构建轨迹
    times = zeros(Float64, num_jumps + 1)
    for i in 1:num_jumps
        # 找到下一个时间
        idx = argmin(next_times)
        times[i + 1] = next_times[idx]

        # 更新下一个时间列表
        next_idxs[idx] += 1
        next_times[idx] = next_idxs[idx] > length(P.times[sidxs[idx]]) ? P.time : P.times[sidxs[idx]][next_idxs[idx]]
    end

    return times
end

"""
    magnetization(P::Plaquettes, idx::Int)

找到一个镶嵌的磁化强度。
"""
function magnetization(P::Plaquettes, idx::Int)
    # 找到时间的重构
    times = reconstructPlaquette(P, idx)

    # 进行时间积分
    Z = 0
    state = deepcopy(P.initial[idx])
    for i in 1:length(times)
        tmin = times[i]
        tmax = i == length(times) ? P.time : times[i + 1]
        Z += (tmax - tmin) * (2 * state - 1)
        state = !state
    end
    return Z
end
