port=8001
# 定义任务配置： "数据集名 序列长度"
configurations=("gsm8k 256")

for j in {1,} 
do
    # 遍历配置数组
    for config in "${configurations[@]}"
    do
        # 读取当前配置 (set -- 将字符串拆分为参数 $1, $2)
        set -- $config
        dataset=$1
        max_seq_len=$2

        for version in "v2" "v3"
        do
            # 1. 这里改为你要测试的 llama 1b 模型
            for model in "meta-llama/llama3.2-1b"
            do
                # 2. 删除原来的 case 判断，新增一个 split_point 的循环 (1到5)
                for current_sp in {1..5}
                do
                    for client_num in 4
                    do
                        for lag_ratios_index in 0 # 0:无异质性
                        do
                            for qo in "fifo"
                            do
                                echo "$j-th Client: Model=$model, Data=$dataset, Len=$max_seq_len, SP=$current_sp, Ver=$version, Clients=$client_num, Port=$port"
                                
                                python experiment/client_run.py \
                                    -NC=${client_num} \
                                    -V=${version} \
                                    -L \
                                    -SP=${current_sp} \
                                    -M=${model} \
                                    -P=${port} \
                                    -B=4 \
                                    -DS=${dataset} \
                                    -SL=${max_seq_len} \
                                    -LAG="${lag_ratios_index}" \
                                    -QO=${qo} 
                                
                                # 端口自增
                                port=$((port + 1))
                                
                                echo "----------------------------------------"
                                # cd vis/
                                # python dcp.py ...
                                # cd ..
                            done
                        done
                    done
                done
            done
        done
    done
done