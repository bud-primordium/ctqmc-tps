#=
    处理从初始状态 si 到终态 sf 在时间 t 内的自旋随机桥的计算和生成。
    它包含一个选项，可以通过某个偏置强度 s 在磁化中引入偏置。
=#

"""
    partition(J::Real, t::Real, i::Bool, f::Bool)

确定在时间 t 内从状态 i 到状态 f 的桥的对数配分函数
"""

function partition(J::Real, t::Real, i::Bool, f::Bool)
    if i == true && f == true
        Z = t * sqrt(1+J^2) - log(2)
        Z += log(1 + (J / sqrt(1 + J^2)) + exp(-2*t*sqrt(1 + J^2)) * (1 - (J / sqrt(1 + J^2))))
    elseif i == false && f == false
        Z = t * sqrt(1+J^2) - log(2)
        Z += log(1 - (J / sqrt(1 + J^2)) + exp(-2*t*sqrt(1 + J^2)) * (1 + (J / sqrt(1 + J^2))))
    else
        Z = t * sqrt(1+J^2) - log(2) - log(sqrt(1 + J^2))
        Z += log(1 -  exp(-2*t*sqrt(1 + J^2)))
    end

    return Z
end


"""
    survival(J::Real, t::Real, tau::Real, i::Bool, f::Bool)

确定在时间 tau 内，从某个初始状态 i 到终态 f 的生存概率，
以及 z 磁化 J 和总时间 t。
"""
function survival(J::Real, t::Real, tau::Real, i::Bool, f::Bool)
    return exp(partition(J, t-tau, i, f) - partition(J, t, i, f) + tau * (i ? J : -J))
end


"""
    survival_time(J::Real, t::Real, i::Bool, f::Bool)   

计算具有磁化强度 J、时间 t、初始状态 i 和终态 f 的系统的生存时间。
"""
function survival_time(J::Real, t::Real, i::Bool, f::Bool)
    # 生成一个随机数来确定时间
    r = rand(Float64)

    # 初始化时间的上下限
    lower = 0
    upper = t

    # 计算整个时间段的配分函数
    Z = partition(J, t, i, f)

    # 检查状态是否存活
    if i == f
        S = exp(partition(J, 0, i, f) - Z + t * (i ? J : -J))
        S > r && return t + 1e-5
    end

    # 二分法寻找目标值
    tau = 0.5*(upper+lower)
    val = exp(partition(J, t-tau, i, f) - Z + tau * (i ? J : -J))
    while abs(val - r) > 1e-8 || isnan(val)
        if val > r && !isnan(val)
            lower = 0.5*(upper+lower)
        else
            upper = 0.5*(upper+lower)
        end
        tau = 0.5*(upper+lower)
        val = exp(partition(J, t-tau, i, f) - Z + tau * (i ? J : -J))
    end

    return tau
end


"""
    bridge(J::Real, t::Real, i::Bool, f::Bool)

从初始状态 i 到终态 f 采样一个桥，给定时间 t 和耦合 J。
"""
function bridge(J::Real, t::Real, i::Bool, f::Bool)
    # 初始化时间
    time = 0.0
    times = Float64[]

    # 找到所有的跃迁时间
    while time < t 
        # 确定生存时间
        dt = survival_time(J, t-time, i, f)

        # 更新系统
        if time + dt < t
            # 存储跃迁
            time += dt
            push!(times, time)
            i = !i
        else
            # 存活
            time += dt
        end
    end
    return times
end
