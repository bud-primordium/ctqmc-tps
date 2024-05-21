using HDF5
using CSV
using DataFrames

function read_specific_data(filesave)
    # 打开HDF5文件
    specific_data = h5open(filesave, "r") do file
        # 获取所有模拟的ID和步数
        sim_ids = keys(file)
        
        # 创建一个空的字典来存储符合条件的数据
        specific_data = Dict{Float64, Dict{String, Any}}()
        
        for sim_id in sim_ids
            steps = keys(file[sim_id])
            for step in steps
                num_sims = read(file, "$sim_id/$step/num_sims")
                if num_sims == 16000
                    J = read(file, "$sim_id/$step/J")
                    specific_data[J] = Dict(
                        "X" => read(file, "$sim_id/$step/X"),
                        "Z" => read(file, "$sim_id/$step/Z"),
                        "Z_var" => read(file, "$sim_id/$step/Z_var"),
                        "num_sims" => num_sims,
                        "J" => J
                    )
                end
            end
        end
        return specific_data
    end
end

function save_to_csv(data, filename)
    # 创建一个 DataFrame 来存储数据
    df = DataFrame(J = Float64[], X = Float64[], Z = Float64[], Z_var = Float64[])
    
    # 按 J 排序并将数据添加到 DataFrame 中
    sorted_keys = sort(collect(keys(data)))
    for J in sorted_keys
        row = data[J]
        push!(df, (J, row["X"], row["Z"], row["Z_var"]))
    end
    
    # 保存 DataFrame 为 CSV 文件
    CSV.write(filename, df)
end

# 读取数据并按J排序
#data = read_specific_data("result at 0.0-0.8.h5")
data = read_specific_data("result at 0.9-1.1-morenums.h5")
#data = read_specific_data("result at 1.2-2.0.h5")

# 保存为CSV文件
#save_to_csv(data, "output-1.csv")
save_to_csv(data, "output-2.csv")
#save_to_csv(data, "output-3.csv")

# 打印排序后的数据
sorted_keys = sort(collect(keys(data)))
for J in sorted_keys
    println("J: ", J)
    println("X: ", data[J]["X"])
    println("Z: ", data[J]["Z"])
    println("Z_var: ", data[J]["Z_var"])
end




