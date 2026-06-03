#!/bin/bash

port=8001
# 与 Client 端保持一致的配置顺序
# configurations=("gsm8k" "dialogsum" "e2e") 
configurations=("gsm8k")
# Server端脚本似乎不需要 max_seq_len (-SL)，如果需要请参照 Client 端修改

for j in {1,}
do
    for dataset in "${configurations[@]}"
    do
        for version in "v2" "v3"
        do
            # 1. 锁定你要测试的模型：llama3.2-1b
            for model in "meta-llama/llama3.2-1b"
            do
                # 2. 删除原有的 case 语句，新增一个 split_point 1到5的循环
                for current_sp in {1..5}
                do
                    for client_num in 4
                    do
                        for lag_ratios_index in 0
                        do
                            for qo in "fifo"
                            do
                                echo "$j-th Server: Model=$model, Data=$dataset, Ver=$version, Clients=$client_num, SP=$current_sp, Port=$port"
                                
                                python experiment/server_run.py \
                                    -NC=${client_num} \
                                    -V=${version} \
                                    -SP=${current_sp} \
                                    -M=${model} \
                                    -P=${port} \
                                    -CKPT='selective' \
                                    -DS=${dataset} \
                                    -LAG="${lag_ratios_index}" \
                                    -QO=${qo}
                                
                                # 端口自增
                                port=$((port + 1))
                            done
                        done
                    done
                done
            done
        done
    done
done